#!/usr/bin/env python3
import argparse
import sys
import logging
import sqlalchemy
import copy
import h5py
import numpy as np
import multiprocessing as mp
from collections import OrderedDict

import CondCore.Utilities.conddblib as conddb

#from conddb
def _inserted_before(_IOV,timestamp):
    '''To be used inside filter().
    '''

    if timestamp is None:
        # XXX: Returning None does not get optimized (skipped) by SQLAlchemy,
        #      and returning True does not work in Oracle (generates "and 1"
        #      which breaks Oracle but not SQLite). For the moment just use
        #      this dummy condition.
        return sqlalchemy.literal(True) == sqlalchemy.literal(True)

    return _IOV.insertion_time <= _parse_timestamp(timestamp)

def _parse_timestamp(timestamp):
    try:
        return datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        pass

    try:
        return datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass

    try:
        return datetime.datetime.strptime(timestamp, '%Y-%m-%d')
    except ValueError:
        pass

    raise Exception("Could not parse timestamp '%s'" % timestamp)

def _exists(session, primary_key, value):
    ret = None
    try: 
        ret = session.query(primary_key).\
            filter(primary_key == value).\
            count() != 0
    except sqlalchemy.exc.OperationalError:
        pass

    return ret

def _connect(db, init, read_only, args, as_admin=False):

    logging.debug('Preparing connection to %s ...', db)

    url = conddb.make_url( db, read_only)
    pretty_url = url
    if url.drivername == 'oracle+frontier':
        ws = url.host.rsplit('%2F')
        if ws is not None:
            pretty_url = 'frontier://%s/%s' %(ws[-1],url.database)
    connTo = '%s [%s]' %(db,pretty_url)
    logging.info('Connecting to %s', connTo)
    logging.debug('DB url: %s',url)
    verbose= 0 
    if args.verbose is not None:
       verbose = args.verbose - 1
    connection = conddb.connect(url, args.authPath, verbose, as_admin)


    if not read_only:
        if connection.is_read_only:
            raise Exception('Impossible to edit a read-only database.')

        if connection.is_official:
            if args.force:
                if not args.yes:
                    logging.warning('You are going to edit an official database. If you are not one of the Offline DB experts but have access to the password for other reasons, please stop now.')
            else:
                raise Exception('Editing official databases is forbidden. Use the official DropBox to upload conditions. If you need a special intervention on the database, see the contact help: %s' % conddb.contact_help)
        # for sqlite we trigger the implicit schema creation
        if url.drivername == 'sqlite':
            if init:
                connection.init()
    if not connection._is_valid:
        raise Exception('No valid schema found in the database.')

    return connection


def connect(args, init=False, read_only=True, as_admin=False):
    args.force = args.force if 'force' in dir(args) else False

    if 'destdb' in args:
        if args.destdb is None:
            args.destdb = args.db
        if args.db == args.destdb:
            conn1 = _connect(args.destdb, init, read_only, args)
            return conn1, conn1
        conn1 = _connect( args.db, init, True, args)
        conn2url = conddb.make_url(args.destdb, False)
        if conn2url.drivername == 'sqlite' and not os.path.exists(args.destdb): 
            init = True
        conn2 = _connect(args.destdb, init, False, args)
        return conn1, conn2

    return _connect( args.db, init, read_only, args, as_admin)


def _high(n):
    return int(n) >> 32

def _low(n):
    return int(n) & 0xffffffff

#end from conddb

#based on conddb._dump_payload
def get_payloads_objtype_data(session, payloads):

    Payload = session.get_dbtype(conddb.Payload)
    table = session.query(Payload.hash, Payload.object_type, Payload.data).\
        filter(Payload.hash.in_(payloads)).order_by(Payload.hash).all()
    return table

def external_process_get_payloads_objtype_data(queue, args, payloads):
    connection = connect(args)
    session = connection.session()
    queue.put(get_payloads_objtype_data(session, payloads))
#local

def timeTypeName(time_type):
    if time_type == conddb.TimeType.Time.value:
        return 'time'
    if time_type == conddb.TimeType.Run.value or time_type == conddb.TimeType.Lumi.value:
        return 'run_lumi'
    raise RuntimeError("unknown since time %s:"% str(time_type))
                       
        

def parseSince(time_type, since):
    if time_type == conddb.TimeType.Time.value:
        return (_high(since), _low(since))
    if time_type == conddb.TimeType.Run.value:
        return (int(since), 0)
    if time_type == conddb.TimeType.Lumi.value:
        return (_high(since), _low(since))

def previousSyncValue(syncValue):
    if syncValue[1] == 0:
        return (syncValue[0]-1, 0xffffffff)
    return (syncValue[0], syncValue[1]-1)
    
def sinceToIOV(sinceList, time_type):
    firstValues = []
    lastValues = []
    for since in sinceList:
        syncValue = parseSince(time_type, since)
        firstValues.append(syncValue)
        if len(firstValues) != 1:
            lastValues.append(previousSyncValue(syncValue))
    lastValues.append((0xFFFFFFFF,0xFFFFFFFF))
    return [firstValues,lastValues]
    
def globalTagInfo(session,name):
    GlobalTag = session.get_dbtype(conddb.GlobalTag)
    GlobalTagMap = session.get_dbtype(conddb.GlobalTagMap)
    try:
        is_global_tag = _exists(session, GlobalTag.name, name)
        if is_global_tag:
            return session.query(GlobalTagMap.record, GlobalTagMap.label, GlobalTagMap.tag_name).\
                filter(GlobalTagMap.global_tag_name == name).\
                order_by(GlobalTagMap.record, GlobalTagMap.label).\
                all()
    except sqlalchemy.exc.OperationalError:
        sys.stderr.write("No table for GlobalTags found in DB.\n\n")
    return None

def tagInfo(session, name, snapshot):
    Tag = session.get_dbtype(conddb.Tag)
    IOV = session.get_dbtype(conddb.IOV)
    is_tag = _exists(session, Tag.name, name)
    if is_tag:
        time_type = session.query(Tag.time_type).\
            filter(Tag.name == name).\
            scalar()
            
        rawTagInfo = session.query(IOV.since, IOV.insertion_time, IOV.payload_hash).\
                     filter(
                         IOV.tag_name == name,
                         _inserted_before(IOV,snapshot),
                     ).\
                     order_by(IOV.since.desc(), IOV.insertion_time.desc()).\
                    from_self().\
                     order_by(IOV.since, IOV.insertion_time).\
                     all()
        filteredTagInfo = []
        lastSince = -1
        for since,insertion,payload in rawTagInfo:
            if lastSince == since:
                continue
            lastSince = since
            filteredTagInfo.append((since,payload))
        return time_type, filteredTagInfo
#                     [sinceLabel, 'Insertion Time', 'Payload', 'Object Type'],
#                     filters = [_since_filter(time_type), None, None, None],
#        )
   
def _checkMerge(previousIOV, newIOV, debugCopy, nExistingDataProducts):
    #sanity check
    #check proper number of entries
    previousSince = -1
    for i,e in enumerate(previousIOV):
        if len(e[1]) != nExistingDataProducts+1:
            raise RuntimeError("entry %i has wrong number of elements %i instead of %i"%(i,len(e[1]),nExistingDataProducts+1))
        if previousSince >= e[0]:
            raise RuntimeError("IOV not in order for index %i"%i)
        previousSince = e[0]

    previousIndex = 0
    debugIndex =0
    while debugIndex < len(debugCopy) and previousIndex < len(previousIOV):
        previousSince = previousIOV[previousIndex][0]
        debugSince = debugCopy[debugIndex][0]
        #print("debugSince: %i, prevSince: %i"%(debugSince,previousSince))
        #print(debugCopy)
        #print(previousIOV)
        if debugSince != previousSince:
            previousIndex +=1
            continue
        if debugCopy[debugIndex][1] != previousIOV[previousIndex][1][:nExistingDataProducts]:
            raise RuntimeError("packaged were not properly copied for index %i original:%s new:%s"%(debugIndex,",".join(debugCopy[debugIndex][1]),",".join(previousIOV[previousIndex][1][:nExistingDataProducts])))
        debugIndex +=1
        previousIndex +=1
    if debugIndex != len(debugCopy):
        raise RuntimeError("failed to copy forward index %i"%debugIndex)
    newIndex = 0
    previousIndex = 0
    while newIndex < len(newIOV) and previousIndex < len(previousIOV):
        previousSince = previousIOV[previousIndex][0]
        newSince = newIOV[newIndex][0]
        if newSince != previousSince:
            previousIndex +=1
            continue
        if previousIOV[previousIndex][1][-1] != newIOV[newIndex][1]:
            raise RuntimeError("failed to append package at index %i"%newIndex)
        previousIndex +=1
        newIndex +=1
    if newIndex != len(newIOV):
        raise RuntimeError("failed to merge IOV entry %i"%newIndex)
    

def mergeIOVs(previousIOV, newIOV):
    debugCopy = copy.deepcopy(previousIOV)
    previousSize = len(previousIOV)
    newSize = len(newIOV)
    previousIndex = 0
    newIndex =0
    nExistingDataProducts = len(previousIOV[0][1])
    while newIndex < newSize and previousIndex < previousSize:

        previousSince = previousIOV[previousIndex][0]
        newSince = newIOV[newIndex][0]
        if previousSince == newSince:
            previousIOV[previousIndex][1].append(newIOV[newIndex][1])
            newIndex +=1
            previousIndex +=1
            continue
        elif newSince < previousSince:
            if previousIndex == 0:
                payloads = [None]*nExistingDataProducts
                payloads.append(newIOV[newIndex][1])
                previousIOV.insert(0,[newSince,payloads])
            else:
                payloads = previousIOV[previousIndex-1][1][:nExistingDataProducts]
                payloads.append(newIOV[newIndex][1])
                previousIOV.insert(previousIndex,[newSince,payloads])
            newIndex +=1
            previousIndex +=1
            previousSize +=1
        elif newSince > previousSince:
            if newIndex == 0:
                previousIOV[previousIndex][1].append(None)
            else:
                if len(previousIOV[previousIndex][1]) == nExistingDataProducts:
                    previousIOV[previousIndex][1].append(newIOV[newIndex-1][1])
            previousIndex +=1
    if newIndex != newSize:
        #print("NEED TO EXTEND")
        #need to append to end
        previousPayloads = previousIOV[-1][1]
        while newIndex != newSize:
            newPayloads = previousPayloads[:]
            newPayloads[nExistingDataProducts] = newIOV[newIndex][1]
            previousIOV.append([newIOV[newIndex][0], newPayloads])
            newIndex +=1
    if previousIndex != previousSize:
        #need to add new item to all remaining entries
        while previousIndex < previousSize:
            previousIOV[previousIndex][1].append(newIOV[-1][1])
            previousIndex +=1
    _checkMerge(previousIOV, newIOV, debugCopy, nExistingDataProducts)
    return previousIOV

def writeTagImpl(tagsGroup, name, time_type, IOV_payloads, payloadToRefs, originalTagNames):
    tagGroup = tagsGroup.create_group(name)
    tagGroup.attrs["time_type"] = timeTypeName(time_type).encode("ascii")
    tagGroup.attrs["db_tags"] = originalTagNames
    firstValues, lastValues = sinceToIOV( (x[0] for x in IOV_payloads), time_type)
    syncValueType = np.dtype([("high", np.uint32),("low", np.uint32)])
    first_np = np.empty(shape=(len(firstValues),), dtype=syncValueType)
    first_np['high'] = [ x[0] for x in firstValues]
    first_np['low'] = [ x[1] for x in firstValues]
    last_np = np.empty(shape=(len(lastValues),), dtype=syncValueType)
    last_np['high'] = [ x[0] for x in lastValues]
    last_np['low'] = [ x[1] for x in lastValues]
    #tagGroup.create_dataset("first",data=np.array(firstValues), dtype=syncValueType)
    #tagGroup.create_dataset("last", data=np.array(lastValues),dtype=syncValueType)
    payloads = [ [ payloadToRefs[y] for y in x[1]] for x in IOV_payloads]
    compressor = None
    if len(first_np) > 100:
        compressor = 'gzip'
    tagGroup.create_dataset("first",data=first_np, compression = compressor)
    tagGroup.create_dataset("last",data=last_np, compression = compressor)
    tagGroup.create_dataset("payload", data=payloads, dtype=h5py.ref_dtype, compression = compressor)
    return tagGroup.ref

    
def writeTag(tagsGroup, time_type, IOV_payloads, payloadToRefs, originalTagNames):
    name = originalTagNames[0]
    if len(originalTagNames) != 1:
        name = name+"@joined"
    return writeTagImpl(tagsGroup, name, time_type, IOV_payloads, payloadToRefs, originalTagNames)
    

def recordToType(record):
    import subprocess
    return subprocess.run(["condRecordToDataProduct",record], capture_output = True, check=True, text=True).stdout

def main():
    parser = argparse.ArgumentParser(description='Read from CMS Condition DB and write to HDF5 file')
    parser.add_argument('--db', '-d', default='pro', help='Database to run the command on. Run the help subcommand for more information: conddb help')
    parser.add_argument('name', nargs='+', help="Name of the global tag.")
    parser.add_argument('--verbose', '-v', action='count', help='Verbosity level. -v prints debugging information of this tool, like tracebacks in case of errors. -vv prints, in addition, all SQL statements issued. -vvv prints, in addition, all results returned by queries.')
    parser.add_argument('--authPath','-a', default=None, help='Path of the authentication .netrc file. Default: the content of the COND_AUTH_PATH environment variable, when specified.')
    parser.add_argument('--snapshot', '-T', default=None, help="Snapshot time. If provided, the output will represent the state of the IOVs inserted into database up to the given time. The format of the string must be one of the following: '2013-01-20', '2013-01-20 10:11:12' or '2013-01-20 10:11:12.123123'.")
    parser.add_argument('--exclude', '-e', nargs='*', help = 'list of records to exclude from the file (can not be used with --include)')
    parser.add_argument('--include', '-i', nargs='*', help = 'lost of the only records that should be included in the file (can not be used with --exclude')
    parser.add_argument('--output', '-o', default='test.h5cond', help='name of hdf5 output file to write')
    args = parser.parse_args()

    if args.exclude and args.include:
        print("Can not use --exclude and --include at the same time")
        exit(-1)

    #build using
    #git grep --cached 'REGISTER_PLUGIN[^F]' | grep -v '\/scripts\/' | grep -v 'registration_macros.h' | grep -v 'HelperMacros.h' | awk -F '(' '{print $2}' | awk -F ')' '{print $1}' > record_to_type
    #recordToType = {}
    #with open("record_to_type", "r") as rtFile:
    #    for l in rtFile:
    #        values = l.split(',')
    #        r =values[0]
    #        t =values[1][1:].strip()
    #        recordToType[r] = t

    #what are key lists??? They seem to hold objects of type 'cond::persistency::KeyList'
    # and have their own proxy type
    keyListRecords = set(["ExDwarfListRcd", "DTKeyedConfigListRcd", "DTKeyedConfigContainerRcd"])
    connection = connect(args)
    session = connection.session()

#    excludeRecords = {"BeamSpotObjectsRcd", "SimBeamSpotObjectsRcd", "BeamSpotOnlineHLTObjectsRcd", "BeamSpotOnlineLegacyObjectsRcd", "DTHVStatusRcd", "EcalLaserAPDPNRatiosRcd"}
    excludeRecords = set()
    if args.exclude:
        excludeRecords = set(args.exclude)
    includeRecords = set()
    if args.include:
        includeRecords = set(args.include)
    
    with h5py.File(args.output, 'w') as h5file:
        recordsGroup = h5file.create_group("Records")
        globalTagsGroup = h5file.create_group("GlobalTags")
        null_dataset = h5file.create_dataset("null_payload", data=np.array([], dtype='b') )
        tagGroupRefs = []
        
        for name in args.name:
            gt = globalTagInfo(session,name)
            lastRcd = None
            lastIOV = []
            dataProductsInRecord = []
            count = 0 #TEMP
            recordDataSize = 0
            for rcd, label, tag in gt:
                if rcd in keyListRecords:
                    continue
                if rcd in excludeRecords:
                    continue
                if includeRecords and (not rcd in includeRecords):
                    continue
                #print("%s %s %s"%(rcd, label, tag))
                #print("value type %s"%(recordToType[rcd],))
                if rcd != lastRcd:
                    if lastRcd is not None:
                        print(" total size:",recordDataSize)
                    recordDataSize = 0
                    print("starting record: ",rcd)
                    if lastRcd is not None:
                        #print("  data products: %s"%(",".join(dataProductsInRecord)))
                        #print("  IOV ",lastIOV)
                        tagGroupRefs.append(writeTag(tagsGroup, time_type, lastIOV, payloadToRefs, originalTagNames))
                    #print("NEW RECORD")
                    recordGroup = recordsGroup.create_group(rcd)
                    tagsGroup = recordGroup.create_group("Tags")
                    dataProductsGroup = recordGroup.create_group("DataProducts")
                    seenPayloads = set()
                    lastRcd = rcd
                    lastIOV = []
                    dataProductsInRecord = []
                    payloadToRefs = { None: null_dataset.ref}
                    originalTagNames = []
                    count +=1 #TEMP

                originalTagNames.append(tag)
                dataProductGroup = dataProductsGroup.create_group(recordToType(rcd)+"@"+label)
                dataProductGroup.attrs["type"] = recordToType(rcd).encode("ascii")
                payloadsGroup = dataProductGroup.create_group("Payloads")
                dataProductsInRecord.append(recordToType(rcd)+"@"+label)
                time_type, iovAndPayload = tagInfo(session, tag, args.snapshot)
                #handle payloads
                # this removes all repeated payloads and preserves the order
                payloadHashs = list(OrderedDict.fromkeys([v[1] for v in iovAndPayload]))
                # also need to remove any already seen
                payloadHashs = [v for v in payloadHashs if v not in seenPayloads]
                #print([v[1] for v in iovAndPayload])
                #print(payloadHashs)
                payloadCache = {}
                payloadHashsIndex = 0
                cacheChunking = 1
                safeChunkingSize = 1
                print(" IOVs: %i"%len(iovAndPayload))
                for index,i_p in enumerate(iovAndPayload):
                    #print("seenPayloads:",seenPayloads)
                    if index % 100 == 0:
                        session.flush()
                        session.commit()
                    if not i_p[1] in seenPayloads:
                        seenPayloads.add(i_p[1])
                        sys.stdout.flush()
                        #print("   %i payload: %s"%(index, i_p[1]))
                        if not payloadCache:
                            #retrieve more info
                            #print("retrieving:",payloadHashs[payloadHashsIndex:payloadHashsIndex+10])
                            cacheChunking = safeChunkingSize
                            queue = mp.Queue()
                            p=mp.Process(target=external_process_get_payloads_objtype_data, args=(queue, args, payloadHashs[payloadHashsIndex:payloadHashsIndex+cacheChunking]))
                            p.start()
                            table = queue.get()
                            p.join()
                            #table = get_payloads_objtype_data(session, payloadHashs[payloadHashsIndex:payloadHashsIndex+cacheChunking])
                            #print(table)
                            payloadHashsIndex +=cacheChunking
                            for r in table:
                                payloadCache[r[0]] = (r[1],r[2])
                        objtype,data = payloadCache[i_p[1]]
                        print("  %i payload: %s size: %i"%(index,i_p[1],len(data)))
                        recordDataSize += len(data)
                        if len(data) < 1000000:
                            safeChunkingSize = 10
                        del payloadCache[i_p[1]]
                        ##print("  cacheSize:",len(payloadCache))
                        #pl = payloadsGroup.create_dataset(i_p[1], data=np.array([1],dtype='b'), compression='gzip')
                        pl = payloadsGroup.create_dataset(i_p[1], data=np.frombuffer(data,dtype='b'), compression='gzip')
                        pl.attrs["type"] = objtype.encode("ascii")
                        payloadToRefs[i_p[1]] = pl.ref
                
                if not lastIOV:
                    lastIOV = [ [i[0],[i[1]]] for i in iovAndPayload]
                else:
                    lastIOV = mergeIOVs(lastIOV, iovAndPayload)
                #print(lenti[1]))
                #if count > 5: #TEMP
                #    break
            print(" total size:",recordDataSize)
            tagGroupRefs.append(writeTag(tagsGroup, time_type, lastIOV, payloadToRefs, originalTagNames))
            globalTagGroup = globalTagsGroup.create_group(name)
            globalTagGroup.create_dataset("Tags", data=tagGroupRefs, dtype=h5py.ref_dtype)
if __name__ == '__main__':
    main()

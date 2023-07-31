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

from CondCore.CondHDF5ESSource.hdf5Writer import writeH5File
import CondCore.Utilities.conddblib as conddb

#Global tags hold a list of Tags
# Tags give the
#      record name,
#      list of data products
#      list of IOVs
#      list of payloads per IOV
# Payloads give
#      a payload name and
#      the serialized data for a data product
#      the type of data for the data product
#



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

class IOVSyncValue(object):
    def __init__(self, high, low):
        self.high = high
        self.low = low

class DBPayloadIterator(object):
    def __init__(self, args, payloads):
        self._args = args
        self._payloadHashs = payloads
        self._payloadCache = {}
        self._payloadHashsIndex = 0
        self._cacheChunking = 1
        self._safeChunkingSize = 1
        self._nextIndex = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self._nextIndex >= len(self._payloadHashs):
            raise StopIteration()
        payloadHash = self._payloadHashs[self._nextIndex]
        if not self._payloadCache:
            self._cacheChunking = self._safeChunkingSize
            queue = mp.Queue()
            p=mp.Process(target=external_process_get_payloads_objtype_data, args=(queue, self._args, self._payloadHashs[self._payloadHashsIndex:self._payloadHashsIndex+self._cacheChunking]))
            p.start()
            table = queue.get()
            p.join()
            #table = get_payloads_objtype_data(session, payloadHashs[payloadHashsIndex:payloadHashsIndex+cacheChunking])
            #print(table)
            self._payloadHashsIndex +=self._cacheChunking
            for r in table:
                self._payloadCache[r[0]] = (r[1],r[2])
        objtype,data = self._payloadCache[payloadHash]
        if len(data) < 1000000:
            self._safeChunkingSize = 10
        del self._payloadCache[payloadHash]
        self._nextIndex +=1
        return DBPayload(payloadHash, canonicalProductName(objtype.encode("ascii")), data)


class DBPayload(object):
    def __init__(self,hash_, type_, data):
        self._hash = hash_
        self._type = type_
        self._data = data
    def name(self):
        return self._hash
    def actualType(self):
        return self._type
    def data(self):
        return self._data

class DBDataProduct(object):
    def __init__(self, ctype, label, payloadHashes, args):
        self._type = ctype
        self._label = label
        self._payloadHashs = payloadHashes
        self._args = args

    def name(self):
        return self._type +"@"+self._label
    def objtype(self):
        return self._type
    def payloads(self):
        return DBPayloadIterator(self._args, self._payloadHashs)

class DBTag(object):
    def __init__(self, session, args, record, productNtags):
        self._session = session
        self._args = args
        self._snapshot = args.snapshot
        self._record = record
        self._productLabels = [x[0] for x in productNtags]
        self._dbtags = [x[1] for x in productNtags]
        self._type = None
        self._iovsNPayloads = None
        self._time_type = None
    def record(self):
        return self._record
    def name(self):
        if len(self._dbtags) == 1:
            return self._dbtags[0]
        return self._dbtags[0]+"@joined"
    def __type(self):
        if self._type is None:
            self._type = recordToType(self._record)
        return self._type
    def time_type(self):
        if self._time_type is None:
            self.iovsNPayloadNames()
        return timeTypeName(self._time_type)
    def originalTagNames(self):
        return self._dbtags
    def iovsNPayloadNames(self):
        if self._iovsNPayloads is None:
            finalIOV = []
            for tag in self._dbtags:
                time_type, iovAndPayload = tagInfo(self._session, tag, self._snapshot)
                self._time_type = time_type
                if not finalIOV:
                    finalIOV = [ [i[0],[i[1]]] for i in iovAndPayload]
                else:
                    finalIOV = mergeIOVs(finalIOV, iovAndPayload)

            firstValues, lastValues = sinceToIOV( (x[0] for x in finalIOV), time_type)

            self._iovsNPayloads = list(zip((IOVSyncValue(x[0],x[1]) for x in firstValues), (IOVSyncValue(x[0], x[1]) for x in lastValues), (x[1] for x in finalIOV)))
            self._session.flush()
            self._session.commit()
        return self._iovsNPayloads

    def dataProducts(self):
        t = self.__type()
        iovs = self.iovsNPayloadNames()
        payloadForProducts = []
        for p in self._productLabels:
            payloadForProducts.append(OrderedDict())
        for first,last,payloads in iovs:
            for i,p in enumerate(payloads):
                if p is not None:
                    payloadForProducts[i][p]=None
        return [DBDataProduct(t,v,list(payloadForProducts[i]), self._args) for i,v in enumerate(self._productLabels)]

class DBGlobalTag(object):
    def __init__(self, args, session, name):
        self._session = session
        self._args = args
        self._snapshot = args.snapshot
        self._name = name
        self._tags = []
        gt = globalTagInfo(session,name)
        lastRcd = None
        tags = []
        for rcd, label, tag in gt:
            if rcd != lastRcd:
                if lastRcd is not None:
                    self._tags.append(DBTag(session,args, lastRcd,tags))
                lastRcd = rcd
                tags = []
            tags.append((label,tag))
        if lastRcd is not None:
            self._tags.append(DBTag(session,args, lastRcd, tags))
    def tags(self):
        return self._tags
            
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
        return (_high(since), 0)
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
            if time_type == conddb.TimeType.Run.value:
                #need to make Run and RunLumi directly comparable since some records
                # use a mix of the two for their IOVs
                since = int(since) << 32
            filteredTagInfo.append((since,payload))

        if time_type == conddb.TimeType.Run.value:
            time_type = conddb.TimeType.Lumi.value

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
            #print(previousIOV,newIOV)
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

def writeTagImpl(tagsGroup, name, recName, time_type, IOV_payloads, payloadToRefs, originalTagNames):
    tagGroup = tagsGroup.create_group(name)
    tagGroup.attrs["time_type"] = time_type.encode("ascii") #timeTypeName(time_type).encode("ascii")
    tagGroup.attrs["db_tags"] = [x.encode("ascii") for x in originalTagNames]
    tagGroup.attrs["record"] = recName.encode("ascii")
    firstValues = [x[0] for x in IOV_payloads]
    lastValues = [x[1] for x in IOV_payloads]
    syncValueType = np.dtype([("high", np.uint32),("low", np.uint32)])
    first_np = np.empty(shape=(len(IOV_payloads),), dtype=syncValueType)
    first_np['high'] = [ x.high for x in firstValues]
    first_np['low'] = [ x.low for x in firstValues]
    last_np = np.empty(shape=(len(lastValues),), dtype=syncValueType)
    last_np['high'] = [ x.high for x in lastValues]
    last_np['low'] = [ x.low for x in lastValues]
    #tagGroup.create_dataset("first",data=np.array(firstValues), dtype=syncValueType)
    #tagGroup.create_dataset("last", data=np.array(lastValues),dtype=syncValueType)
    payloads = [ [ payloadToRefs[y] for y in x[2]] for x in IOV_payloads]
    compressor = None
    if len(first_np) > 100:
        compressor = 'gzip'
    tagGroup.create_dataset("first",data=first_np, compression = compressor)
    tagGroup.create_dataset("last",data=last_np, compression = compressor)
    tagGroup.create_dataset("payload", data=payloads, dtype=h5py.ref_dtype, compression = compressor)
    return tagGroup.ref

    
def writeTag(tagsGroup, time_type, IOV_payloads, payloadToRefs, originalTagNames, recName):
    name = originalTagNames[0]
    if len(originalTagNames) != 1:
        name = name+"@joined"
    return writeTagImpl(tagsGroup, name, recName, time_type, IOV_payloads, payloadToRefs, originalTagNames)
    

def recordToType(record):
    import subprocess
    return subprocess.run(["condRecordToDataProduct",record], capture_output = True, check=True, text=True).stdout

__typedefs = {b"ESCondObjectContainer<ESPedestal>":"ESPedestals",
              b"ESCondObjectContainer<float>":"ESFloatCondObjectContainer",
              b"ESCondObjectContainer<ESChannelStatusCode>":"ESChannelStatus",
              b"EcalCondObjectContainer<EcalPedestal>":"EcalPedestals",
              b"EcalCondObjectContainer<EcalXtalGroupId>":"EcalWeightXtalGroups",
              b"EcalCondObjectContainer<EcalMGPAGainRatio>":"EcalGainRatios",
              b"EcalCondObjectContainer<float>":"EcalFloatCondObjectContainer",
              b"EcalCondObjectContainer<EcalChannelStatusCode>":"EcalChannelStatus",
              b"EcalCondObjectContainer<EcalMappingElement>":"EcalMappingElectronics",
              b"EcalCondObjectContainer<EcalTPGPedestal>":"EcalTPGPedestals",
              b"EcalCondObjectContainer<EcalTPGLinearizationConstant>":"EcalTPGLinearizationConst",
              b"EcalCondObjectContainer<EcalTPGCrystalStatusCode>":"EcalTPGCrystalStatus",
              b"EcalCondTowerObjectContainer<EcalChannelStatusCode>":"EcalDCSTowerStatus",
              b"EcalCondTowerObjectContainer<EcalDAQStatusCode>":"EcalDAQTowerStatus",
              b"EcalCondObjectContainer<EcalDQMStatusCode>":"EcalDQMChannelStatus",
              b"EcalCondTowerObjectContainer<EcalDQMStatusCode>":"EcalDQMTowerStatus",
              b"EcalCondObjectContainer<EcalPulseShape>":"EcalPulseShapes",
              b"EcalCondObjectContainer<EcalPulseCovariance>":"EcalPulseCovariances",
              b"EcalCondObjectContainer<EcalPulseSymmCovariance>":"EcalPulseSymmCovariances",
              b"HcalItemCollById<HFPhase1PMTData>": "HFPhase1PMTParams",
              b"l1t::CaloParams":"CaloParams",
              b"StorableDoubleMap<AbsOOTPileupCorrection>":"OOTPileupCorrectionMapColl",
              b"PhysicsTools::Calibration::Histogram3D<double,double,double,double>":"PhysicsTools::Calibration::HistogramD3D",
              b"PhysicsTools::Calibration::MVAComputerContainer":"MVAComputerContainer"
}
def canonicalProductName(product):
    return __typedefs.get(product,product)

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
    parser.add_argument('--compressor', '-c', default='zlib', choices =['zlib','lzma','none'], help="compress data using 'zlib', 'lzma' or 'none'")    
    args = parser.parse_args()

    if args.exclude and args.include:
        print("Can not use --exclude and --include at the same time")
        exit(-1)

    connection = connect(args)
    session = connection.session()

    excludeRecords = set()
    if args.exclude:
        excludeRecords = set(args.exclude)
    includeRecords = set()
    if args.include:
        includeRecords = set(args.include)

    writeH5File(args.output, args.name, excludeRecords, includeRecords, lambda x: DBGlobalTag(args, session,  x), args.compressor )
    
if __name__ == '__main__':
    main()

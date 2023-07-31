#!/usr/bin/env python3
import argparse
import sys
import logging
import copy
import h5py
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
import zlib

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


class IOVSyncValue(object):
    def __init__(self, high, low):
        self.high = high
        self.low = low

class H5Payload(object):
    def __init__(self,dataset,name):
        self._dataset = dataset
        self._hash = name
        self._type = dataset.attrs['type']
    def name(self):
        return self._hash
    def actualType(self):
        return self._type
    def data(self):
        return self._dataset[()]

class H5DataProduct(object):
    def __init__(self, group, name):
        self._type = group.attrs['type']
        self._name = name
        self._payloadGroup = group['Payloads']
    def name(self):
        return self._name
    def objtype(self):
        return self._type
    def payloads(self):
        return [H5Payload(self._payloadGroup[p],p.split('/')[-1]) for p in self._payloadGroup]
    def idToPayloadNames(self):
        return { self._payloadGroup[p].id:p.split('/')[-1] for p in self._payloadGroup }

class H5Tag(object):
    def __init__(self, file, group, name):
        self._file = file
        self._group = group
        self._record = self._group.attrs['record']
        self._name = name

        recordGroup = file['Records'][self._record]
        dataProductsGroup = recordGroup['DataProducts']

        self._dataProducts = [H5DataProduct(dataProductsGroup[g],g.split('/')[-1]) for g in dataProductsGroup]
        self._dbtags = self._group.attrs['db_tags']
        self._time_type = self._group.attrs['time_type']
    def record(self):
        return self._record
    def name(self):
        return self._name
    def time_type(self):
        return self._time_type
    def originalTagNames(self):
        return self._dbtags
    def iovsNPayloadNames(self):

        #asking an h5 object for its name is a slow operation
        idToName = {self._file['null_payload'].id: None}
        for d in self._dataProducts:
            idToName.update(d.idToPayloadNames())

        first = self._group['first'][()]
        last = self._group['last'][()]
        payloadRefs = self._group['payload']
        return list(zip( (IOVSyncValue(x['high'],x['low']) for x in first),
                         (IOVSyncValue(x['high'],x['low']) for x in last),
                         ([idToName[self._file[r].id] for r in refs] for refs in payloadRefs)) )

    def dataProducts(self):
        return self._dataProducts

class H5GlobalTag(object):
    def __init__(self, filename, name):
        self._file = h5py.File(filename,'r')
        self._name = name
                
    def tags(self):
        #looking up names is slow so better to make cache
        tagID2Name = {}
        recordsGroup = self._file['Records']
        for recordName in recordsGroup:
            r = recordsGroup[recordName]
            tagsGroup = r['Tags']
            for tagName in tagsGroup:
                tagID2Name[tagsGroup[tagName].id] = tagName
        globalTagGroup = self._file['GlobalTags'][self._name]
        return [H5Tag(self._file, self._file[t], tagID2Name[self._file[t].id]) for t in globalTagGroup['Tags']]

            

def writeTagImpl(tagsGroup, name, recName, time_type, IOV_payloads, payloadToRefs, originalTagNames):
    tagGroup = tagsGroup.create_group(name)
    tagGroup.attrs["time_type"] = time_type.encode("ascii")
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
    
def main():
    parser = argparse.ArgumentParser(description='Read from HDF5 file and write to HDF5 file')
    parser.add_argument('input', help="Name of file to read")
    parser.add_argument('name', nargs='+', help="Name of the global tag.")

    parser.add_argument('--exclude', '-e', nargs='*', help = 'list of records to exclude from the file (can not be used with --include)')
    parser.add_argument('--include', '-i', nargs='*', help = 'lost of the only records that should be included in the file (can not be used with --exclude')
    parser.add_argument('--output', '-o', default='test.h5cond', help='name of hdf5 output file to write')
    args = parser.parse_args()

    if args.exclude and args.include:
        print("Can not use --exclude and --include at the same time")
        exit(-1)

    #what are key lists??? They seem to hold objects of type 'cond::persistency::KeyList'
    # and have their own proxy type
    keyListRecords = set(["ExDwarfListRcd", "DTKeyedConfigListRcd", "DTKeyedConfigContainerRcd"])

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
            gt = H5GlobalTag(args.input, name)
            for tag in gt.tags():
                rcd = tag.record()
                if rcd in keyListRecords:
                    continue
                if rcd in excludeRecords:
                    continue
                if includeRecords and (not rcd in includeRecords):
                    continue
                recordDataSize = 0
                
                payloadToRefs = { None: null_dataset.ref}
                
                recordGroup = recordsGroup.create_group(rcd)
                tagsGroup = recordGroup.create_group("Tags")
                dataProductsGroup = recordGroup.create_group("DataProducts")
                print("record: %s"%rcd)
                for dataProduct in tag.dataProducts():
                    dataProductGroup = dataProductsGroup.create_group(dataProduct.name())
                    dataProductGroup.attrs["type"] = dataProduct.objtype().encode("ascii")
                    payloadsGroup = dataProductGroup.create_group("Payloads")
                    print(" product: %s"%dataProduct.name())
                    for p_index, payload in enumerate(dataProduct.payloads()):
                        print("  %i payload: %s size: %i"%(p_index,payload.name(),len(payload.data())))
                        recordDataSize +=len(payload.data())
                        b = zlib.compress(payload.data())
                        if len(b) >= len(payload.data()):
                            #compressing isn't helping
                            b = payload.data()
                        pl = payloadsGroup.create_dataset(payload.name(), data=np.frombuffer(b,dtype='b'))
                        pl.attrs["memsize"] = len(payload.data())
                        pl.attrs["type"] = payload.actualType()
                        payloadToRefs[payload.name()] = pl.ref
                        
                tagGroupRefs.append(writeTag(tagsGroup, tag.time_type(), tag.iovsNPayloadNames(), payloadToRefs, tag.originalTagNames(), rcd))
                print(" total size:",recordDataSize)
                recordDataSize = 0

            globalTagGroup = globalTagsGroup.create_group(name)
            globalTagGroup.create_dataset("Tags", data=tagGroupRefs, dtype=h5py.ref_dtype)
if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import sys
import logging
import copy
import h5py
import numpy as np
from collections import OrderedDict
import zlib
import lzma
from CondCore.CondHDF5ESSource.hdf5Writer import writeH5File

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
    def __init__(self,dataset,name, compressor):
        self._dataset = dataset
        self._hash = name
        self._type = dataset.attrs['type']
        self._memsize = dataset.attrs['memsize']
        self._compressor = compressor
    def name(self):
        return self._hash
    def actualType(self):
        return self._type
    def memsize(self):
        return self._memsize
    def data(self):
        ds = self._dataset[()]
        if len(ds) == self.memsize():
            return ds
        #was compressed
        return self._compressor.decompress(ds)

class H5DataProduct(object):
    def __init__(self, group, name, compressor):
        self._type = group.attrs['type']
        self._name = name
        self._payloadGroup = group['Payloads']
        self._compressor = compressor
    def name(self):
        return self._name
    def objtype(self):
        return self._type
    def payloads(self):
        return [H5Payload(self._payloadGroup[p],p.split('/')[-1], self._compressor) for p in self._payloadGroup]
    def idToPayloadNames(self):
        return { self._payloadGroup[p].id:p.split('/')[-1] for p in self._payloadGroup }

class H5Tag(object):
    def __init__(self, file, group, name):
        self._file = file

        compressor = None
        compressorName = self._file.attrs['default_payload_compressor']
        if compressorName == 'lzma':
            compressor = lzma
        if compressorName == 'zlib':
            compressor = zlib
        self._group = group
        self._record = self._group.attrs['record']
        self._name = name

        recordGroup = file['Records'][self._record]
        dataProductsGroup = recordGroup['DataProducts']

        self._dataProducts = [H5DataProduct(dataProductsGroup[g],g.split('/')[-1], compressor) for g in dataProductsGroup]
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
        return (H5Tag(self._file, self._file[t], tagID2Name[self._file[t].id]) for t in globalTagGroup['Tags'])

def main():
    parser = argparse.ArgumentParser(description='Read from HDF5 file and write to HDF5 file')
    parser.add_argument('input', help="Name of file to read")
    parser.add_argument('name', nargs='+', help="Name of the global tag.")

    parser.add_argument('--exclude', '-e', nargs='*', help = 'list of records to exclude from the file (can not be used with --include)')
    parser.add_argument('--include', '-i', nargs='*', help = 'lost of the only records that should be included in the file (can not be used with --exclude')
    parser.add_argument('--output', '-o', default='test.h5cond', help='name of hdf5 output file to write')
    parser.add_argument('--compressor', '-c', default='zlib', choices=['zlib', 'lzma', 'none'], help="compress data using 'zlib', 'lzma' or 'none'")
    args = parser.parse_args()

    if args.exclude and args.include:
        print("Can not use --exclude and --include at the same time")
        exit(-1)

    excludeRecords = set()
    if args.exclude:
        excludeRecords = set(args.exclude)
    includeRecords = set()
    if args.include:
        includeRecords = set(args.include)

    writeH5File(args.output, args.name, excludeRecords, includeRecords, lambda x: H5GlobalTag(args.input, x),  args.compressor)
if __name__ == '__main__':
    main()

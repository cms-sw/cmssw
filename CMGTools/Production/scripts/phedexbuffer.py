#!/bin/env python

import json
import pprint
import operator
import urllib2 

def teraByte( byte ):
    return float(byte) / 1024. / 1024. / 1024. / 1024.

class DataSet(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def __str__(self):
        tmp = '{size:3.1f}  {name} '.format( size=self.size,
                                             name=self.name )
        return tmp

def fetchData():
    print 'accessing phedex on cmsweb.cern.ch'
    url = 'https://cmsweb.cern.ch/phedex/datasvc/json/prod/subscriptions?node=T2_CH_CERN&group=local&create_since=0'
    h1 = urllib2.urlopen( url ) 
    content = h1.read()
    # jsondata = res.read()
    print 'saving json data'
    save = open('save.json','w')
    save.write( content )


if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = "%prog [options]\nPhedex buffer status."
    parser.add_option("-c", "--cache", dest="useCache", default=False,
                                  action='store_true',
                                  help='read status from the cache.')

    (options,args) = parser.parse_args()

    if len(args)!=0:
        parser.print_help()
        sys.exit(1)


    readCache = options.useCache
    if not readCache:
        fetchData()

    res = None
    try:
        res = open('save.json')
    except IOError:
        fetchData()
        res = open('save.json')
        
    objs = json.loads( res.read() )
    datasets = objs['phedex']['dataset']

    myDatasets = []

    totSize = 0
    for dataset in datasets:
        name = dataset['name']
        size = teraByte(dataset['bytes'])
        totSize += size
        myds = DataSet( name, size ) 
        myDatasets.append( myds )
        # print myds

    # pprint.pprint(datasets[0])
    print 'by name:'
    for ds in sorted(myDatasets, key=operator.attrgetter('name') ):
        print ds
    print 
    print 'by size'
    for ds in sorted(myDatasets, key=operator.attrgetter('size') ):
        print ds
    print
    print 'TOTAL:', totSize, ' TB'

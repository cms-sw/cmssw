#!/usr/bin/env python
from __future__ import print_function
VERSION='2.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,lumiCalcAPI,sessionManager

def listRemoveDuplicate(inlist):
    d={}
    for x in inlist:
        d[x]=x
    return d.values()

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="list lumi data availability")
    # add required arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    # add optional arguments
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',
                        default=None,
                        required=False,
                        help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('action',choices=['listrun'],help='command actions')
    parser.add_argument('--minrun',
                        dest='minrun',
                        action='store',
                        type=int,
                        help='min run number')
    parser.add_argument('--maxrun',
                        dest='maxrun',
                        action='store',
                        type=int,
                        help='max run number')
    parser.add_argument('--minfill',
                        dest='minfill',
                        type=int,
                        action='store',
                        help='min fill number')
    parser.add_argument('--maxfill',
                        dest='maxfill',
                        type=int,
                        action='store',
                        help='max fill number')
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help='verbose mode for printing' )
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    # parse arguments
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    reqTrg=True
    reqHLT=False
    session.transaction().start(True)
    schema=session.nominalSchema()
    runlist=lumiCalcAPI.runList(schema,None,runmin=options.minrun,runmax=options.maxrun,startT=None,stopT=None,l1keyPattern=None,hltkeyPattern=None,amodetag=None,nominalEnergy=None,energyFlut=None,requiretrg=reqTrg,requirehlt=reqHLT)
    session.transaction().commit()
    if options.action == 'listrun':
        if not runlist:
            print('[]')
            sys.exit(0)
        singlelist=sorted(listRemoveDuplicate(runlist))
        print(singlelist)
    del session
    del svc
if __name__=='__main__':
    main()


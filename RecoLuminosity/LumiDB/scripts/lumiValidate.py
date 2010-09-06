#!/usr/bin/env python
VERSION='1.00'
import os,sys
import coral
import json
#import optparse
from RecoLuminosity.LumiDB import inputFilesetParser,selectionParser,argparse,CommonUtil
import RecoLuminosity.LumiDB.lumiQueryAPI as LumiQueryAPI

#from pprint import pprint

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi Validation",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['batchupdate','update']
    allowedFlags = ['UNKNOWN','GOOD','BAD','SUSPECT']
    # parse arguments
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='path to authentication file')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file,required for batchupdate action')
    parser.add_argument('-o',dest='outputfile',action='store',help='output to csv file')
    parser.add_argument('-runls',dest='runls',action='store',help='selection string,optional. Example [1234:[],4569:[1,1],[2,100]]')
    parser.add_argument('-flag',dest='flag',action='store',default='UNKNOWN',help='flag string,optional')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode for printing' )
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    options=parser.parse_args()
    if options.flag.upper() not in allowedFlags:
        print 'unrecognised flag ',options.flag.upper()
        raise
    os.environ['CORAL_AUTH_PATH'] = options.authpath
    connectstring=options.connect
    svc = coral.ConnectionService()
    session=svc.connect(connectstring,accessMode=coral.access_Update)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")

    if options.debug :
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    if options.action=='batchupdate':
        #populate from csv file, require -i argument
        if not options.inputfile:
            print 'inputfile -i option is required for batchupdate'
            raise
        fparser=inputFilesetParser.inputFilesetParser(options.inputfile)
        runsandls=fparser.runsandls()
    if options.action=='update':
        #update flag interactively, require -runls argument
        #runls={run:[]} populate all CMSLSNUM found in LUMISUMMARY
        #runls={run:[[1,1],[2,5]],run:[[1,1],[2,5]]}
        #default value
        if not options.runls:
            print 'option -runls is required for update'
            raise
        runlsjson=CommonUtil.tolegalJSON(options.runls)
        sparser=selectionParser.selectionParser(runlsjson)
        runsandls=sparser.runsandls()
    print runsandls
    del session
    del svc
if __name__ == '__main__':
    main()

    


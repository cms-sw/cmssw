#!/usr/bin/env python
#
# update lumisummary set instlumi=instlumi*:norm where runnum>=:startrun and runnum<=:endrun
#
#
# update lumisummary set instlumi=instlumi*:norm where runnum=:run 
#
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse
def calibrateRange(dbsession,normfactor,startrun,endrun):
    '''
    update lumisummary set instlumi=instlumi*:norm where runnum>=:startrun and runnum<=:endrun
    '''
    try:
        dbsession.transaction().start(False)
        schema=dbsession.nominalSchema()
        if not schema:
            raise 'cannot connect to schema'
        if not schema.existsTable('LUMISUMMARY'):
            raise 'non-existing table LUMISUMMARY'
        inputData=coral.AttributeList()
        inputData.extend('normfactor','float')
        inputData['normfactor'].setData(float(normfactor))
        inputData.extend('startrun','unsigned int')
        inputData['startrun'].setData(int(startrun))
        inputData.extend('endrun','unsigned int')
        inputData['endrun'].setData(int(endrun))
        nchanged=schema.tableHandle('LUMISUMMARY').dataEditor().updateRows('INSTLUMI=INSTLUMI*:normfactor','RUNNUM>=:startrun AND RUNNUM<=:endrun',inputData)
        dbsession.transaction().commit()
        return nchanged
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
def calibrateRun(dbsession,normfactor,runnum):
    '''
    update lumisummary set instlumi=instlumi*:norm where runnum=:run 
    '''
    try:
        dbsession.transaction().start(False)
        schema=dbsession.nominalSchema()
        if not schema:
            raise 'cannot connect to schema'
        if not schema.existsTable('LUMISUMMARY'):
            raise 'non-existing table LUMISUMMARY'
        inputData=coral.AttributeList()
        inputData.extend('normfactor','float')
        inputData['normfactor'].setData(float(normfactor))
        inputData.extend('runnumber','unsigned int')
        inputData['runnumber'].setData(int(runnum))
        nchanged=schema.tableHandle('LUMISUMMARY').dataEditor().updateRows('INSTLUMI=INSTLUMI*:normfactor','RUNNUM=:runnumber',inputData)
        dbsession.transaction().commit()
        return nchanged
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="apply normalization factor to inst lumi")
    parser.add_argument('-c',dest='connectstr',action='store',required=True,help='connectstr')
    parser.add_argument('-norm',dest='normfactor',action='store',required=True,help='normalization factor to apply')
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',required=False,help='run number')
    parser.add_argument('-startrun',dest='startrun',action='store',required=False,help='start run for range action')
    parser.add_argument('-endrun',dest='endrun',action='store',required=False,help='end run for range action')
    parser.add_argument('action',choices=['run','range'],help='calibrate run')
    
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    connectstr=args.connectstr
    normfactor=args.normfactor
    if len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    else:
        raise '-P authpath argument is required'
    svc=coral.ConnectionService()
    session=svc.connect(connectstr,accessMode=coral.access_Update)
    if args.debug:
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    n=0
    if args.action == 'run':
        runnumber=0
        if args.runnumber:
            runnumber=args.runnumber
        else:
            raise 'argument -r is required for action run'
        if args.debug:
            print 'connectstr : ',connectstr
            print 'normfactor : ',normfactor
            print 'authpath : ',os.environ['CORAL_AUTH_PATH']
            print 'runnumber : ',runnumber
        n=calibrateRun(session,normfactor,runnumber)
    if args.action == 'range':
        startrun=0
        endrun=0
        if args.startrun:
            startrun=args.startrun
        else:
            raise 'argument -startrun is required for action range'
        if args.endrun:
            endrun=args.endrun
        else:
            raise 'argument -endrun is required for action range'
        if args.debug:
            print 'connectstr : ',connectstr
            print 'normfactor : ',normfactor
            print 'authpath : ',os.environ['CORAL_AUTH_PATH']
            print 'startrun : ',startrun
            print 'endrun : ',endrun
        n=calibrateRange(session,normfactor,startrun,endrun)
    print 'number of rows changed: ',n
    del session
    del svc
        
if __name__=='__main__':
    main()
    

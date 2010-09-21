#!/usr/bin/env python
VERSION='1.02'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse

class constants(object):
    def __init__(self):
        self.debug=False
        #self.norbits=2**18
        #self.nbx=3564
        self.gtmondb=''
        #self.gtmondb='oracle://cms_omds_lb/CMS_GT_MON'
        self.gtmonschema='CMS_GT_MON'
        self.deadviewname='GT_MON_TRIG_DEAD_VIEW'
        self.algoviewname='GT_MON_TRIG_ALGO_VIEW'
        
def bitzeroForRun(dbsession,c,runnum):
    '''
    select lsnr,counts,prescale from CMS_GT_MON.GT_MON_TRIG_ALGO_VIEW where runnr=:runnumber and algobit=:bitnum order by lsnr
    '''
    result={}
    try:
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.gtmonschema)
        if not schema:
            raise 'cannot connect to schema',c.gtmonschema
        if not schema.existsView(c.algoviewname):
            raise 'non-existing view',c.algoviewname

        bitOutput=coral.AttributeList()
        bitOutput.extend("lsnr","unsigned int")
        bitOutput.extend("algocount","unsigned int")
        bitBindVarList=coral.AttributeList()
        bitBindVarList.extend("runnumber","unsigned int")
        bitBindVarList.extend("bitnum","unsigned int")
        bitBindVarList["runnumber"].setData(int(runnum))
        bitBindVarList["bitnum"].setData(0)
        
        query=schema.newQuery()
        query.addToTableList(c.algoviewname)
        query.addToOutputList('LSNR','lsnr')
        query.addToOutputList('COUNTS','algocount')
        query.setCondition('RUNNR=:runnumber AND ALGOBIT=:bitnum',bitBindVarList)
        query.addToOrderList('lsnr')
        query.defineOutput(bitOutput)
        
        cursor=query.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['lsnr'].data()
            algocount=cursor.currentRow()['algocount'].data()
            result[cmslsnum]=algocount
        del query
        dbsession.transaction().commit()
        #print result
        return result
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
    
def deadcountForRun(dbsession,c,runnum):
    '''
    select counts,lsnr from cms_gt_mon.gt_mon_trig_dead_view where runnr=:runnumber and deadcounter=:countername order by lsnr;
    '''
    result={}
    try:
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.gtmonschema)
        if not schema:
            raise 'cannot connect to schema',c.gtmonschema
        if not schema.existsView(c.deadviewname):
            raise 'non-existing view',c.deadviewname

        deadOutput=coral.AttributeList()
        deadOutput.extend("lsnr","unsigned int")
        deadOutput.extend("deadcount","unsigned long long")
        
        deadBindVarList=coral.AttributeList()
        deadBindVarList.extend("runnumber","unsigned int")
        deadBindVarList.extend("countername","string")
        deadBindVarList["runnumber"].setData(int(runnum))
        deadBindVarList["countername"].setData('DeadtimeBeamActive')
        
        query=schema.newQuery()
        query.addToTableList(c.deadviewname)
        query.addToOutputList('LSNR','lsnr')
        query.addToOutputList('COUNTS','deadcount')
        query.setCondition('RUNNR=:runnumber AND DEADCOUNTER=:countername',deadBindVarList)
        query.addToOrderList('lsnr')
        query.defineOutput(deadOutput)
        
        cursor=query.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['lsnr'].data()
            deadcount=cursor.currentRow()['deadcount'].data()
            result[cmslsnum]=deadcount
        del query
        dbsession.transaction().commit()
        return result
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def main():
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Dump GT info")
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to trigger DB(required)')
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',required=True,help='run number')
    parser.add_argument('action',choices=['deadtime','deadfraction'],help='dump deadtime beamacrive count')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=args.runnumber
    c.gtmondb=args.connect
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    svc=coral.ConnectionService()
    session=svc.connect(c.gtmondb,accessMode=coral.access_ReadOnly)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    if args.debug:
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    
    if args.action == 'deadtime':
        deadresult=deadcountForRun(session,c,runnumber)
        if deadresult and len(deadresult)!=0:
            print 'run',runnumber
            print 'ls deadcount'
            for cmsls,deadcount in deadresult.items():
                print cmsls,deadcount
        else:
            print 'no deadtime found for run ',runnumber
            
    if args.action == 'deadfraction':
        deadresult=deadcountForRun(session,c,runnumber)
        bitzeroresult=bitzeroForRun(session,c,runnumber)
        if deadresult and len(deadresult)!=0:
            print 'run',runnumber
            print 'ls deadfraction'
            for cmsls,deadcount in deadresult.items():
                bitzero_count=bitzeroresult[cmsls]
                bitzero_prescale=1.0
                if int(runnumber)>=146315:
                    bitzero_prescale=17.0
                if bitzero_count==0:
                    print cmsls,'no beam'
                else:
                    print cmsls,'%.5f'%float(float(deadcount)/(float(bitzero_count)*bitzero_prescale))
        else:
            print 'no deadtime found for run ',runnumber
        
    del session
    del svc
        
if __name__=='__main__':
    main()
    

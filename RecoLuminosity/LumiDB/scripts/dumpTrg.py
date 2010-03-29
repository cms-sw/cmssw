#!/usr/bin/env python
VERSION='1.02'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse

class constants(object):
    def __init__(self):
        self.debug=False
        self.norbits=2**18
        self.nbx=3564
        self.gtmondb='oracle://cms_omds_lb/CMS_GT_MON'
        self.gtmonschema='CMS_GT_MON'
        self.deadviewname='GT_MON_TRIG_DEAD_VIEW'

def deadfracForRun(dbsession,c,runnum):
    '''
    select counts/(NORBITS*NBX),lsnr from cms_gt_mon.gt_mon_trig_dead_view where runnr=:runnumber and deadcounter=:countername order by lsnr;
    '''
    result=[]
    try:
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.gtmonschema)
        if not schema:
            raise 'cannot connect to schema',c.gtmonschema
        if not schema.existsView(c.deadviewname):
            raise 'non-existing view',c.deadviewname

        deadOutput=coral.AttributeList()
        deadOutput.extend("lsnr","unsigned int")
        deadOutput.extend("deadfrac","float")
        
        deadBindVarList=coral.AttributeList()
        deadBindVarList.extend("runnumber","unsigned int")
        deadBindVarList.extend("countername","string")
        deadBindVarList["runnumber"].setData(int(runnum))
        deadBindVarList["countername"].setData('Deadtime')
        
        query=schema.newQuery()
        query.addToTableList(c.deadviewname)
        query.addToOutputList('LSNR','lsnr')
        query.addToOutputList('COUNTS/('+str(c.norbits)+'*'+str(c.nbx)+')','deadcount')
        query.setCondition('RUNNR=:runnumber AND DEADCOUNTER=:countername',deadBindVarList)
        query.addToOrderList('lsnr')
        query.defineOutput(deadOutput)
        
        cursor=query.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['lsnr'].data()
            deadfrac=cursor.currentRow()['deadfrac'].data()
            result.append((cmslsnum,deadfrac))
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
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',required=True,help='run number')
    parser.add_argument('action',choices=['deadfrac'],help='dump deadfraction')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=args.runnumber
    
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    svc=coral.ConnectionService()
    session=svc.connect(c.gtmondb,accessMode=coral.access_ReadOnly)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    if args.debug:
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    
    if args.action == 'deadfrac':
        deadresult=deadfracForRun(session,c,runnumber)
        if deadresult and len(deadresult)!=0:
            print 'run',runnumber
            print 'ls deadfraction'
            for (cmsls,deadfrac) in deadresult:
                print cmsls,'%.3f'%deadfrac
        else:
            print 'no deadtime found for run ',runnumber
    del session
    del svc
        
if __name__=='__main__':
    main()
    

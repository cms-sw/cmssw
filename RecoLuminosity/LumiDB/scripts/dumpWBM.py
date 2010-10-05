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
        self.wbmdb=''
        self.wbmschema='CMS_WBM'
        self.deadtable='LEVEL1_TRIGGER_CONDITIONS'
        self.algotable='LEVEL1_TRIGGER_ALGO_CONDITIONS'
        self.techtable='LEVEL1_TRIGGER_TECH_CONDITIONS'
def bitzeroForRun(dbsession,c,runnum):
    '''
      select LUMISEGMENTNR,GTALGOCOUNTS from CMS_WBM.LEVEL1_TRIGGER_ALGO_CONDITIONS where RUNNUMBER=:runnumber and BIT=0 order by LUMISEGMENTNR
    '''
    result={}
    try:
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.wbmschema)
        if not schema:
            raise Exception,'cannot connect to schema'+c.wbmschema
        if not schema.existsTable(c.algotable):
            raise Exception,'non-existing view'+c.algotable
        
        query=schema.newQuery()
        algoBindVarList=coral.AttributeList()
        algoBindVarList.extend("runnumber","unsigned int")
        algoBindVarList.extend("bitnumber","unsigned int")
        algoBindVarList["runnumber"].setData(int(runnum))
        algoBindVarList["bitnumber"].setData(int(0))
        query.addToTableList(c.algotable)
        query.addToOutputList("LUMISEGMENTNR","lsnr")
        query.addToOutputList('GTALGOCOUNTS','bitcount')
        query.setCondition('RUNNUMBER=:runnumber AND BIT=:bitnumber',algoBindVarList)
        query.addToOrderList('LUMISEGMENTNR')

        bitzeroOutput=coral.AttributeList()
        bitzeroOutput.extend("lsnr","unsigned int")
        bitzeroOutput.extend('bitcount','unsigned int')
        
        query.defineOutput(bitzeroOutput)
        cursor=query.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['lsnr'].data()
            bitcount=cursor.currentRow()['bitcount'].data()
            result[cmslsnum]=bitcount
        del query
        dbsession.transaction().commit()
        return result
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession

def deadcountForRun(dbsession,c,runnum):
    '''
    select DEADTIMEBEAMACTIVE from cms_wbm.LEVEL1_TRIGGER_CONDITIONS where RUNNUMBER=133881 order by LUMISEGMENTNR
    '''
    result={}
    try:   
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.wbmschema)
        if not schema:
            raise Exception,'cannot connect to schema '+c.wbmschema
        if not schema.existsTable(c.deadtable):
            raise Exception,'non-existing table '+c.deadtable

        deadOutput=coral.AttributeList()
        deadOutput.extend("lsnr","unsigned int")
        deadOutput.extend("deadcount","unsigned long long")
        
        deadBindVarList=coral.AttributeList()
        deadBindVarList.extend("RUNNUMBER","unsigned int")
        deadBindVarList["RUNNUMBER"].setData(int(runnum))
        
        query=schema.newQuery()
        query.addToTableList(c.deadtable)
        query.addToOutputList('LUMISEGMENTNR','lsnr')
        query.addToOutputList('DEADTIMEBEAMACTIVE','deadcount')
        query.setCondition('RUNNUMBER=:runnumber',deadBindVarList)
        query.addToOrderList('LUMISEGMENTNR')
        query.defineOutput(deadOutput)
        
        cursor=query.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['lsnr'].data()
            deadcount=cursor.currentRow()['deadcount'].data()
            result[cmslsnum]=deadcount
            #print 'deadcount',deadcount
        del query
        dbsession.transaction().commit()
        return result
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def main():
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Dump trigger info in wbm")
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to trigger DB(required)')
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',required=True,help='run number')
    parser.add_argument('action',choices=['deadtime','deadfraction'],help='dump deadfraction')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=args.runnumber
    c.wbmdb=args.connect
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    #print 'authpath ',args.authpath
    svc=coral.ConnectionService()
    #print 'about to access ',c.wbmdb
    session=svc.connect(c.wbmdb,accessMode=coral.access_ReadOnly)
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
        print 'run',runnumber
        print 'ls deadfraction'
        if deadresult and len(deadresult)!=0:
            #print 'run',runnumber
            #print 'ls deadfraction'
            for cmsls,deadcount in deadresult.items():
                bitzero=bitzeroresult[cmsls]
                bitzero_prescale=1.0
                if int(runnumber)>=146315:
                    bitzero_prescale=17.0
                if bitzero==0:
                    print cmsls,'no beam'
                else:
                    print cmsls,'%.5f'%float(float(deadcount)/(float(bitzero)*bitzero_prescale))
        else:
            print 'no deadtime found for run ',runnumber
    del session
    del svc
        
if __name__=='__main__':
    main()
    

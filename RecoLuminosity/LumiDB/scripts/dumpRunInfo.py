#!/usr/bin/env python
VERSION='1.02'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse

class constants(object):
    def __init__(self):
        self.debug=False
        self.runinfodb=''
        self.runinfoschema='CMS_RUNINFO'
        self.runsessionparameterTable='RUNSESSION_PARAMETER'
        self.hltconfname='CMS.LVL0:HLT_KEY_DESCRIPTION'
        
def hltkeyForRun(dbsession,c,runnum):
    '''
    select runnumber,string_value from cms_runinfo.runsession_parameter where name=:runsessionparametername and runnumber=:runnum 
    '''
    result={}
    try:
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.runinfoschema)
        if not schema:
            raise Exception, 'cannot connect to schema '+c.runinfoschema
        if not schema.existsTable(c.runsessionparameterTable):
            raise Exception, 'non-existing table '+c.runsessionparameterTable

        hltkeyOutput=coral.AttributeList()
        hltkeyOutput.extend("runnum","unsigned int")
        hltkeyOutput.extend("hltkey","string")
        
        bindVarList=coral.AttributeList()
        bindVarList.extend("name","string")
        bindVarList.extend("runnumber","unsigned int")

        bindVarList["name"].setData(c.hltconfname)
        bindVarList["runnumber"].setData(int(runnum))
        
        query=schema.newQuery()
        query.addToTableList(c.runsessionparameterTable)
        query.addToOutputList('RUNNUMBER','runnumber')
        query.addToOutputList('STRING_VALUE','value')
        query.setCondition('NAME=:name AND RUNNUMBER=:runnumber',bindVarList)
        query.defineOutput(hltkeyOutput)
        
        cursor=query.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            hltkey=cursor.currentRow()['hltkey'].data()
            result[runnum]=hltkey
        del query
        dbsession.transaction().commit()
        #print result
        return result
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
    
def main():
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Dump Run info")
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to trigger DB(required)')
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',required=True,help='run number')
    parser.add_argument('action',choices=['hltkey'],help='dump hltconf key')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=args.runnumber
    c.runinfodb=args.connect
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    svc=coral.ConnectionService()
    session=svc.connect(c.runinfodb,accessMode=coral.access_ReadOnly)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    if args.debug:
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    
    if args.action == 'hltkey':
        result=hltkeyForRun(session,c,runnumber)
        print 'runnumber hltkey'
        for runnum,hltkey in result.items():
            print runnum,hltkey

    del session
    del svc
        
if __name__=='__main__':
    main()
    

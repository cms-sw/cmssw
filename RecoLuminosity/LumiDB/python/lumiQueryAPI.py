import os
import coral
from RecoLuminosity.LumiDB import nameDealer,lumiTime
'''
This module defines lowlevel SQL query API for lumiDB 
We do not like range queries so far because of performance of range scan. 
The principle is to query by runnumber and per each coral queryhandle
Try reuse db session/transaction and just renew query handle each time to reduce metadata queries.
Avoid explicit order by, should be solved by asc index in the schema.
Do not handle transaction in here.
Do not do explicit del in here.
'''
def runsummaryByrun(queryHandle,runnum):
    '''
    select fillnum,sequence,hltkey,to_char(starttime),to_char(stoptime) from cmsrunsummary where runnum=:runnum
    output: [fillnum,sequence,hltkey,starttime,stoptime]
    '''
    t=lumiTime.lumiTime()
    result=[]
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName(),'runsummary')
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('runsummary.fillnum')
    queryHandle.addToOutputList('runsummary.sequence')
    queryHandle.addToOutputList('runsummary.hltkey')
    queryHandle.addToOutputList('to_char(runsummary.starttime,\''+t.coraltimefm+'\')')
    queryHandle.addToOutputList('to_char(runsummary.stoptime,\''+t.coraltimefm+'\')')
    queryHandle.setCondition('runnum=:runnum',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('fillnum','unsigned int')
    queryResult.extend('sequence','string')
    queryResult.extend('hltkey','string')
    queryResult.extend('starttime','string')
    queryResult.extend('stoptime','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        result.append(cursor.currentRow()['fillnum'].data())
        result.append(cursor.currentRow()['sequence'].data())
        result.append(cursor.currentRow()['hltkey'].data())
        result.append(cursor.currentRow()['starttime'].data())
        result.append(cursor.currentRow()['stoptime'].data())
    if len(result)!=5:
        print 'wrong runsummary result'
        raise
    return result

def lumisummaryByrun(queryHandle,runnum,lumiversion):
    '''
    select cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenery from lumisummary where runnum=:runnum and lumiversion=:lumiversion
    output: {cmslsnum:[instlumi,numorbit,startorbit,beamstatus,beamenergy]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.cmslumisummaryTableName(),'lumisummary')
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('lumiversion','string')
    
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['lumiversion'].setData(lumiversion)
    queryHandle.addToOutputList('lumisummary.cmslsnum')
    queryHandle.addToOutputList('lumisummary.instlumi')
    queryHandle.addToOutputList('lumisummary.numorbit')
    queryHandle.addToOutputList('lumisummary.startorbit')
    queryHandle.addToOutputList('lumisummary.beamstatus')
    queryHandle.addToOutputList('lumisummary.beamenergy')
    queryHandle.setCondition('runnum=:runnum and lumiversion=:lumiversion',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('instlumi','float')
    queryResult.extend('numorbit','unsigned int')
    queryResult.extend('startorbit','unsigned int')
    queryResult.extend('beamstatus','string')
    queryResult.extend('beamenergy','float')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        instlumi=cursor.currentRow()['instlumi'].data()
        numorbit=cursor.currentRow()['numorbit'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        beamstatus=cursor.currentRow()['beamstatus'].data()
        beamenergy=cursor.currentRow()['beamenergy'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[instlumi,numorbit,startorbit,beamstatus,beamenergy]
    return result

def lumisumByrun(queryHandle,runnum,lumiversion):
    '''
    select sum(instlumi) from lumisummary where runnum=:runnum and lumiversion=:lumiversion
    output: float totallumi
    Note: the output is the raw result, need to apply LS length in time(sec)
    '''
    pass

def trgbitzeroByrun(queryHandle,runnum):
    '''
    select cmslsnum,trgcount,deadtime,bitname,prescale from trg where runnum=:runnum and bitnum=0;
    output: {cmslsnum:[trgcount,deadtime,bitname,prescale]}
    '''
    pass

def lumisummarytrgbitzeroByrun(queryHandle,runnum,lumiversion):
    '''
    Everything you need to know about bitzero. Since we do not know if joint query is better of sperate. So support both.
    output: {cmslsnum:[instlumi,numorbit,startorbit,beamstatus,beamenergy,bitzerocount,deadtime,bitname,prescale]}
    '''
    pass

def trgBybitnameByrun(queryHandle,runnum,bitname):
    '''
    select cmslsnum,trgcount,deadtime,bitnum,prescale from trg where runnum=:runnum and bitname=:bitname;
    output: {cmslsnum:[trgcount,deadtime,bitnum,prescale]}
    '''
    pass

def trgAllbitsByrun(queryHandle,runnum):
    '''
    select cmslsnum,trgcount,deadtime,bitnum,bitname,prescale from trg where runnum=:runnum
    this can be changed to blob query later
    output: {cmslsnum:{bitname:[bitnum,trgcount,deadtime,prescale]}}
    '''
    pass

def hltBypathByrun(queryHandle,runnum,hltpath):
    '''
    select cmslsnum,inputcount,acceptcount,prescale from hlt where runnum=:runnum and pathname=:hltpath
    output: {cmslsnum:[inputcount,acceptcount,prescale]}
    '''
    pass

def hltAllpathByrun(queryHandle,runum):
    '''
    select cmslsnum,inputcount,acceptcount,prescale,hltpath from hlt where runnum=:runnum
    this can be changed to blob query later
    output: {cmslsnum:{hltpath:[inputcount,acceptcount,prescale]}}
    '''
    pass

def lumidetailByrunByAlgo(queryHandle,runum,algoname='OCC1'):
    '''
    select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnumber and d.algoname=:algoname and s.lumisummary_id=d.lumisummary_id order by s.startorbit
    output: [cmslsnum,bxlumivalue,bxlumierror,bxlumiquality,startorbit]
    since the output is ordered by time, it has to be in seq list format
    '''
    pass

def lumidetailAllalgosByrun(queryHandle,runum):
    '''
    select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,d.algoname from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnumber and s.lumisummary_id=d.lumisummary_id order by s.startorbit
    output: {algoname:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality,startorbit]}
    '''
    pass

def hlttrgMappingByrun(queryHandle,runnum):
    '''
    select trghltmap.hltpathname,trghltmap.l1seed from cmsrunsummary,trghltmap where cmsrunsummary.runnum=:runnum and trghltmap.hltkey=cmsrunsummary.hltkey
    output: {hltpath:l1seed}
    '''
    pass

if __name__=='__main__':
    msg=coral.MessageStream('')
    #msg.setMsgVerbosity(coral.message_Level_Debug)
    msg.setMsgVerbosity(coral.message_Level_Error)
    os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/cms/DB/lumi'
    svc = coral.ConnectionService()
    connectstr='oracle://cms_orcoff_prod/cms_lumi_prod'
    session=svc.connect(connectstr,accessMode=coral.access_ReadOnly)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    session.transaction().start(True)
    schema=session.nominalSchema()
    q=schema.newQuery()
    runsummaryOut=runsummaryByrun(q,139400)
    session.transaction().commit()  
    del q
    del session
    del svc
    print runsummaryOut

import os
import coral,datetime
from RecoLuminosity.LumiDB import nameDealer,lumiTime
'''
This module defines lowlevel SQL query API for lumiDB 
We do not like range queries so far because of performance of range scan.Use only necessary.
The principle is to query by runnumber and per each coral queryhandle
Try reuse db session/transaction and just renew query handle each time to reduce metadata queries.
Avoid unnecessary explicit order by, mostly solved by asc index in the schema.
Do not handle transaction in here.
Do not do explicit del queryhandle in here.
Note: all the returned dict format are not sorted by itself.Sort it outside if needed.
'''
def runsummaryByrun(queryHandle,runnum):
    '''
    select fillnum,sequence,hltkey,to_char(starttime),to_char(stoptime) from cmsrunsummary where runnum=:runnum
    output: [fillnum,sequence,hltkey,starttime,stoptime]
    '''
    t=lumiTime.lumiTime()
    result=[]
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('FILLNUM','fillnum')
    queryHandle.addToOutputList('SEQUENCE','sequence')
    queryHandle.addToOutputList('HLTKEY','hltkey')
    queryHandle.addToOutputList('to_char(STARTTIME,\''+t.coraltimefm+'\')','starttime')
    queryHandle.addToOutputList('to_char(STOPTIME,\''+t.coraltimefm+'\')','stoptime')
    queryHandle.setCondition('RUNNUM=:runnum',queryCondition)
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
    select cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenery from lumisummary where runnum=:runnum and lumiversion=:lumiversion order by startorbit;
    output: [[cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenergy,cmsalive]]
    Note: the non-cmsalive LS are included in the result
    '''
    result=[]
    queryHandle.addToTableList(nameDealer.lumisummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('lumiversion','string')
    
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['lumiversion'].setData(lumiversion)
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('INSTLUMI','instlumi')
    queryHandle.addToOutputList('NUMORBIT','numorbit')
    queryHandle.addToOutputList('STARTORBIT','startorbit')
    queryHandle.addToOutputList('BEAMSTATUS','beamstatus')
    queryHandle.addToOutputList('BEAMENERGY','beamenergy')
    queryHandle.addToOutputList('CMSALIVE','cmsalive')
    queryHandle.setCondition('RUNNUM=:runnum and LUMIVERSION=:lumiversion',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('instlumi','float')
    queryResult.extend('numorbit','unsigned int')
    queryResult.extend('startorbit','unsigned int')
    queryResult.extend('beamstatus','string')
    queryResult.extend('beamenergy','float')
    queryResult.extend('cmsalive','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        instlumi=cursor.currentRow()['instlumi'].data()
        numorbit=cursor.currentRow()['numorbit'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        beamstatus=cursor.currentRow()['beamstatus'].data()
        beamenergy=cursor.currentRow()['beamenergy'].data()
        cmsalive=cursor.currentRow()['cmsalive'].data()
        result.append([cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenergy,cmsalive])
    return result

def lumisumByrun(queryHandle,runnum,lumiversion,beamstatus=None,beamenergy=None,beamenergyfluctuation=0.09):
    '''
    beamenergy unit : GeV
    beamenergyfluctuation : fraction allowed to fluctuate around beamenergy value
    select sum(instlumi) from lumisummary where runnum=:runnum and lumiversion=:lumiversion
    output: float totallumi
    Note: the output is the raw result, need to apply LS length in time(sec)
    '''
    result=0.0
    queryHandle.addToTableList(nameDealer.lumisummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('lumiversion','string')
    
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['lumiversion'].setData(lumiversion)
    queryHandle.addToOutputList('sum(INSTLUMI)','lumitotal')
    conditionstring='RUNNUM=:runnum and LUMIVERSION=:lumiversion'
    if beamstatus:
        conditionstring=conditionstring+' and BEAMSTATUS=:beamstatus'
        queryCondition.extend('beamstatus','string')
        queryCondition['beamstatus'].setData(beamstatus)
    if beamenergy:
        minBeamenergy=float(beamenergy*(1-beamenergyfluctuation))
        maxBeamenergy=float(beamenergy*(1+beamenergyfluctuation))
        conditionstring=conditionstring+' and BEAMENERGY>:minBeamenergy and BEAMENERGY<:maxBeamenergy'
        queryCondition.extend('minBeamenergy','float')
        queryCondition.extend('maxBeamenergy','float')
        queryCondition['minBeamenergy'].setData(float(minBeamenergy))
        queryCondition['maxBeamenergy'].setData(float(maxBeamenergy))
    queryHandle.setCondition(conditionstring,queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('lumitotal','float')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        result=cursor.currentRow()['lumitotal'].data()
    return result

def trgbitzeroByrun(queryHandle,runnum):
    '''
    select cmslsnum,trgcount,deadtime,bitname,prescale from trg where runnum=:runnum and bitnum=0;
    output: {cmslsnum:[trgcount,deadtime,bitname,prescale]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.trgTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('bitnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['bitnum'].setData(int(0))
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('TRGCOUNT','trgcount')
    queryHandle.addToOutputList('DEADTIME','deadtime')
    queryHandle.addToOutputList('BITNAME','bitname')
    queryHandle.addToOutputList('PRESCALE','prescale')
    queryHandle.setCondition('RUNNUM=:runnum and BITNUM=:bitnum',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('trgcount','unsigned int')
    queryResult.extend('deadtime','unsigned int')
    queryResult.extend('bitname','string')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        trgcount=cursor.currentRow()['trgcount'].data()
        deadtime=cursor.currentRow()['deadtime'].data()
        bitname=cursor.currentRow()['bitname'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[trgcount,deadtime,bitname,prescale]
    return result

def lumisummarytrgbitzeroByrun(queryHandle,runnum,lumiversion):
    '''
    select l.cmslsnum,l.instlumi,l.numorbit,l.startorbit,l.beamstatus,l.beamenery,t.trgcount,t.deadtime,t.bitname,t.prescale from trg t,lumisummary l where t.bitnum=:bitnum and l.runnum=:runnum and l.lumiversion=:lumiversion and l.runnum=t.runnum and t.cmslsnum=l.cmslsnum; 
    Everything you ever need to know about bitzero and avg luminosity. Since we do not know if joint query is better of sperate, support both.
    output: {cmslsnum:[instlumi,numorbit,startorbit,beamstatus,beamenergy,bitzerocount,deadtime,bitname,prescale]}
    Note: only cmsalive LS are included in the result. Therefore, this function cannot be used for calculating delivered!
    '''
    result={}
    queryHandle.addToTableList(nameDealer.trgTableName(),'t')
    queryHandle.addToTableList(nameDealer.lumisummaryTableName(),'l')
    queryCondition=coral.AttributeList()
    queryCondition.extend('bitnum','unsigned int')
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('lumiversion','string')
    queryCondition['bitnum'].setData(int(0))        
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['lumiversion'].setData(lumiversion)
    queryHandle.addToOutputList('l.CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('l.INSTLUMI','instlumi')
    queryHandle.addToOutputList('l.NUMORBIT','numorbit')
    queryHandle.addToOutputList('l.STARTORBIT','startorbit')
    queryHandle.addToOutputList('l.BEAMSTATUS','beamstatus')
    queryHandle.addToOutputList('l.BEAMENERGY','beamenergy')
    queryHandle.addToOutputList('t.TRGCOUNT','trgcount')
    queryHandle.addToOutputList('t.DEADTIME','deadtime')
    queryHandle.addToOutputList('t.BITNAME','bitname')
    queryHandle.addToOutputList('t.PRESCALE','prescale')
    queryHandle.setCondition('t.BITNUM=:bitnum and l.RUNNUM=:runnum and l.LUMIVERSION=:lumiversion and l.RUNNUM=t.RUNNUM and t.CMSLSNUM=l.CMSLSNUM',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('instlumi','float')
    queryResult.extend('numorbit','unsigned int')
    queryResult.extend('startorbit','unsigned int')
    queryResult.extend('beamstatus','string')
    queryResult.extend('beamenergy','float')  
    queryResult.extend('trgcount','unsigned int')
    queryResult.extend('deadtime','unsigned int')
    queryResult.extend('bitname','string')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        instlumi=cursor.currentRow()['instlumi'].data()
        numorbit=cursor.currentRow()['numorbit'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        beamstatus=cursor.currentRow()['beamstatus'].data()
        beamenergy=cursor.currentRow()['beamenergy'].data()
        trgcount=cursor.currentRow()['trgcount'].data()
        deadtime=cursor.currentRow()['deadtime'].data()
        bitname=cursor.currentRow()['bitname'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[instlumi,numorbit,startorbit,beamstatus,beamenergy,trgcount,deadtime,bitname,prescale]
    return result

def trgBybitnameByrun(queryHandle,runnum,bitname):
    '''
    select cmslsnum,trgcount,deadtime,bitnum,prescale from trg where runnum=:runnum and bitname=:bitname;
    output: {cmslsnum:[trgcount,deadtime,bitnum,prescale]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.trgTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('bitname','string')
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['bitname'].setData(bitname)        
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('TRGCOUNT','trgcount')
    queryHandle.addToOutputList('DEADTIME','deadtime')
    queryHandle.addToOutputList('BITNUM','bitnum')
    queryHandle.addToOutputList('PRESCALE','prescale')
    queryHandle.setCondition('RUNNUM=:runnum and BITNAME=:bitname',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('trgcount','unsigned int')
    queryResult.extend('deadtime','unsigned long long')
    queryResult.extend('bitnum','unsigned int')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        trgcount=cursor.currentRow()['trgcount'].data()
        deadtime=cursor.currentRow()['deadtime'].data()
        bitnum=cursor.currentRow()['bitnum'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[trgcount,deadtime,bitnum,prescale]
    return result

def trgAllbitsByrun(queryHandle,runnum):
    '''
    all you ever want to know about trigger
    select cmslsnum,trgcount,deadtime,bitnum,bitname,prescale from trg where runnum=:runnum order by  bitnum,cmslsnum
    this can be changed to blob query later
    output: {cmslsnum:{bitname:[bitnum,trgcount,deadtime,prescale]}}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.trgTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('cmslsnum')
    queryHandle.addToOutputList('trgcount')
    queryHandle.addToOutputList('deadtime')
    queryHandle.addToOutputList('bitnum')
    queryHandle.addToOutputList('bitname')
    queryHandle.addToOutputList('prescale')
    queryHandle.setCondition('runnum=:runnum',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('trgcount','unsigned int')
    queryResult.extend('deadtime','unsigned long long')
    queryResult.extend('bitnum','unsigned int')
    queryResult.extend('bitname','string')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    queryHandle.addToOrderList('bitnum')
    queryHandle.addToOrderList('cmslsnum')
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        trgcount=cursor.currentRow()['trgcount'].data()
        deadtime=cursor.currentRow()['deadtime'].data()
        bitnum=cursor.currentRow()['bitnum'].data()
        bitname=cursor.currentRow()['bitname'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            dataperLS={}
            dataperLS[bitname]=[bitnum,trgcount,deadtime,prescale]
            result[cmslsnum]=dataperLS
        else:
            result[cmslsnum][bitname]=[bitnum,trgcount,deadtime,prescale]
    return result


def hltBypathByrun(queryHandle,runnum,hltpath):
    '''
    select cmslsnum,inputcount,acceptcount,prescale from hlt where runnum=:runnum and pathname=:pathname
    output: {cmslsnum:[inputcount,acceptcount,prescale]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.hltTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('pathname','string')
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['pathname'].setData(hltpath)
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('INPUTCOUNT','inputcount')
    queryHandle.addToOutputList('ACCEPTCOUNT','acceptcount')
    queryHandle.addToOutputList('PRESCALE','prescale')
    queryHandle.setCondition('RUNNUM=:runnum and PATHNAME=:pathname',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('inputcount','unsigned int')
    queryResult.extend('acceptcount','unsigned int')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        inputcount=cursor.currentRow()['inputcount'].data()
        acceptcount=cursor.currentRow()['acceptcount'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[inputcount,acceptcount,prescale]
    return result

def hltAllpathByrun(queryHandle,runnum):
    '''
    select cmslsnum,inputcount,acceptcount,prescale,pathname from hlt where runnum=:runnum
    this can be changed to blob query later
    output: {cmslsnum:{pathname:[inputcount,acceptcount,prescale]}}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.hltTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('INPUTCOUNT','inputcount')
    queryHandle.addToOutputList('ACCEPTCOUNT','acceptcount')
    queryHandle.addToOutputList('PRESCALE','prescale')
    queryHandle.addToOutputList('PATHNAME','pathname')
    queryHandle.setCondition('RUNNUM=:runnum',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('inputcount','unsigned int')
    queryResult.extend('acceptcount','unsigned int')
    queryResult.extend('prescale','unsigned int')
    queryResult.extend('pathname','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        inputcount=cursor.currentRow()['inputcount'].data()
        acceptcount=cursor.currentRow()['acceptcount'].data()
        prescale=cursor.currentRow()['prescale'].data()
        pathname=cursor.currentRow()['pathname'].data()
        if not result.has_key(cmslsnum):
            dataperLS={}
            dataperLS[pathname]=[inputcount,acceptcount,prescale]
            result[cmslsnum]=dataperLS
        else:
            result[cmslsnum][pathname]=[inputcount,acceptcount,prescale]
    return result

def lumidetailByrunByAlgo(queryHandle,runnum,algoname='OCC1'):
    '''
    select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,s.startorbit from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnum and d.algoname=:algoname and s.lumisummary_id=d.lumisummary_id order by s.startorbit
    output: [[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality,startorbit]]
    since the output is ordered by time, it has to be in seq list format
    '''
    result=[]
    queryHandle.addToTableList(nameDealer.lumidetailTableName(),'d')
    queryHandle.addToTableList(nameDealer.lumisummaryTableName(),'s')
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('algoname','string')
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['algoname'].setData(algoname)
    queryHandle.addToOutputList('s.CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('d.BXLUMIVALUE','bxlumivalue')
    queryHandle.addToOutputList('d.BXLUMIERROR','bxlumierror')
    queryHandle.addToOutputList('d.BXLUMIQUALITY','bxlumiquality')
    queryHandle.addToOutputList('s.STARTORBIT','startorbit')
    queryHandle.setCondition('s.runnum=:runnum and d.algoname=:algoname and s.lumisummary_id=d.lumisummary_id',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('bxlumivalue','blob')
    queryResult.extend('bxlumierror','blob')
    queryResult.extend('bxlumiquality','blob')
    queryResult.extend('startorbit','unsigned int')    
    queryHandle.addToOrderList('s.STARTORBIT')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
        bxlumierror=cursor.currentRow()['bxlumierror'].data()
        bxlumiquality=cursor.currentRow()['bxlumiquality'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        result.append([cmslsnum,bxlumivalue,bxlumierror,bxlumiquality,startorbit])
    return result

def lumidetailAllalgosByrun(queryHandle,runnum):
    '''
    select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,d.algoname,s.startorbit from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnumber and s.lumisummary_id=d.lumisummary_id order by s.startorbit,d.algoname
    output: {algoname:{cmslsnum:[bxlumivalue,bxlumierror,bxlumiquality,startorbit]}}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.lumidetailTableName(),'d')
    queryHandle.addToTableList(nameDealer.lumisummaryTableName(),'s')
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('s.CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('d.BXLUMIVALUE','bxlumivalue')
    queryHandle.addToOutputList('d.BXLUMIERROR','bxlumierror')
    queryHandle.addToOutputList('d.BXLUMIQUALITY','bxlumiquality')
    queryHandle.addToOutputList('d.ALGONAME','algoname')
    queryHandle.addToOutputList('s.STARTORBIT','startorbit')
    queryHandle.setCondition('s.RUNNUM=:runnum and s.LUMISUMMARY_ID=d.LUMISUMMARY_ID',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('bxlumivalue','blob')
    queryResult.extend('bxlumierror','blob')
    queryResult.extend('bxlumiquality','blob')
    queryResult.extend('algoname','string')
    queryResult.extend('startorbit','unsigned int')    
    queryHandle.addToOrderList('startorbit')
    queryHandle.addToOrderList('algoname')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
        bxlumierror=cursor.currentRow()['bxlumierror'].data()
        bxlumiquality=cursor.currentRow()['bxlumiquality'].data()
        algoname=cursor.currentRow()['algoname'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        if not result.has_key(algoname):
            dataPerAlgo={}
            dataPerAlgo[cmslsnum]=[bxlumivalue,bxlumierror,bxlumiquality,startorbit]
            result[algoname]=dataPerAlgo
        else:
            result[algoname][cmslsnum]=[bxlumivalue,bxlumierror,bxlumiquality,startorbit]           
    return result

def hlttrgMappingByrun(queryHandle,runnum):
    '''
    select m.hltpathname,m.l1seed from cmsrunsummary r,trghltmap m where r.runnum=:runnum and m.hltkey=r.hltkey
    output: {hltpath:l1seed}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName(),'r')
    queryHandle.addToTableList(nameDealer.trghltMapTableName(),'m')
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('m.HLTPATHNAME','hltpathname')
    queryHandle.addToOutputList('m.L1SEED','l1seed')
    queryHandle.setCondition('r.RUNNUM=:runnum and m.HLTKEY=r.HLTKEY',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('hltpathname','string')
    queryResult.extend('l1seed','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        hltpathname=cursor.currentRow()['hltpathname'].data()
        l1seed=cursor.currentRow()['l1seed'].data()
        if not result.has_key(hltpathname):
            result[hltpathname]=l1seed
    return result

def runsByfillrange(queryHandle,minFill,maxFill):
    '''
    find all runs in the fill range inclusive
    select runnum,fillnum from cmsrunsummary where fillnum>=:minFill and fillnum<=:maxFill
    output: fillDict={fillnum:[runlist]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('minFill','unsigned int')
    queryCondition.extend('maxFill','unsigned int')
    queryCondition['minFill'].setData(int(minFill))
    queryCondition['maxFill'].setData(int(maxFill))
    queryHandle.addToOutputList('RUNNUM','runnum')
    queryHandle.addToOutputList('FILLNUM','fillnum')
    queryHandle.setCondition('FILLNUM>=:minFill and FILLNUM<=:maxFill',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('runnum','unsigned int')
    queryResult.extend('fillnum','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        runnum=cursor.currentRow()['runnum'].data()
        fillnum=cursor.currentRow()['fillnum'].data()
        if not result.has_key(fillnum):
            result[fillnum]=[runnum]
        else:
            result[fillnum].append(runnum)
    return result

def runsByTimerange(queryHandle,minTime,maxTime):
    '''
    find all runs in the time range inclusive
    the selected run must have started after minTime and finished by maxTime
    select runnum,to_char(startTime),to_char(stopTime) from cmsrunsummary where startTime>=timestamp(minTime) and stopTime<=timestamp(maxTime);
    input: minTime,maxTime in python obj datetime.datetime
    output: {runnum:[starttime,stoptime]} return in python obj datetime.datetime
    '''
    t=lumiTime.lumiTime()
    result={}
    coralminTime=coral.TimeStamp(minTime.year,minTime.month,minTime.day,minTime.hour,minTime.minute,minTime.second,0)
    coralmaxTime=coral.TimeStamp(maxTime.year,maxTime.month,maxTime.day,maxTime.hour,maxTime.minute,maxTime.second,0)
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('minTime','time stamp')
    queryCondition.extend('maxTime','time stamp')
    queryCondition['minTime'].setData(coralminTime)
    queryCondition['maxTime'].setData(coralmaxTime)
    queryHandle.addToOutputList('RUNNUM','runnum')
    queryHandle.addToOutputList('TO_CHAR(STARTTIME,\''+t.coraltimefm+'\')','starttime')
    queryHandle.addToOutputList('TO_CHAR(STOPTIME,\''+t.coraltimefm+'\')','stoptime')
    queryHandle.setCondition('STARTTIME>=:minTime and STOPTIME<=:maxTime',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('runnum','unsigned int')
    queryResult.extend('starttime','string')
    queryResult.extend('stoptime','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        runnum=cursor.currentRow()['runnum'].data()
        starttimeStr=cursor.currentRow()['starttime'].data()
        stoptimeStr=cursor.currentRow()['stoptime'].data()
        if not result.has_key(runnum):
            result[runnum]=[t.StrToDatetime(starttimeStr),t.StrToDatetime(stoptimeStr)]
    return result
    
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
    del q
    q=schema.newQuery()
    lumisummaryOut=lumisummaryByrun(q,139400,'0001')
    del q
    q=schema.newQuery()
    lumitotal=lumisumByrun(q,139400,'0001')
    del q
    q=schema.newQuery()
    lumitotalStablebeam7TeV=lumisumByrun(q,139400,'0001',beamstatus='STABLE BEAMS',beamenergy=3.5E003,beamenergyfluctuation=0.09)
    del q
    q=schema.newQuery()
    trgbitzero=trgbitzeroByrun(q,139400)
    del q
    q=schema.newQuery()
    lumijointrg=lumisummarytrgbitzeroByrun(q,139400,'0001')
    del q
    q=schema.newQuery()
    trgforbit=trgBybitnameByrun(q,139400,'L1_ZeroBias')
    del q
    q=schema.newQuery()
    trgallbits=trgAllbitsByrun(q,139400)
    del q
    q=schema.newQuery()
    hltbypath=hltBypathByrun(q,139400,'HLT_Mu5')
    del q
    q=schema.newQuery()
    hltallpath=hltAllpathByrun(q,139400)
    del q
    q=schema.newQuery()
    hlttrgmap=hlttrgMappingByrun(q,139400)
    del q
    q=schema.newQuery()
    occ1detail=lumidetailByrunByAlgo(q,139400,'OCC1')
    del q
    q=schema.newQuery()
    alldetail=lumidetailAllalgosByrun(q,139400)
    del q
    q=schema.newQuery()
    runsbyfill=runsByfillrange(q,1150,1170)
    del q
    now=datetime.datetime.now()
    aweek=datetime.timedelta(weeks=1)
    lastweek=now-aweek
    print lastweek
    q=schema.newQuery()
    runsinaweek=runsByTimerange(q,lastweek,now)
    del q
    session.transaction().commit()  
    del session
    del svc
    print 'runsummaryByrun : ',runsummaryOut
    print
    print 'lumisummaryByrun : ',lumisummaryOut
    print
    print 'totallumi : ',lumitotal
    print
    print
    print 'totallumi stable beam and 7TeV: ',lumitotalStablebeam7TeV
    print
    #print 'trgbitzero : ',trgbitzero
    print 
    #print 'lumijointrg', lumijointrg
    print
    #print 'trgforbit L1_ZeroBias ',trgforbit
    print
    #print 'trgallbits ',trgallbits[1] #big query. be aware of speed
    print
    print 'hltforpath HLT_Mu5',hltbypath
    print
    print 'hltallpath ',hltallpath
    print
    print 'hlttrgmap ',hlttrgmap
    print
    print 'lumidetail occ1 ',len(occ1detail)
    print
    print 'runsbyfill ',runsbyfill
    print
    print 'runsinaweek ',runsinaweek.keys()
    
    

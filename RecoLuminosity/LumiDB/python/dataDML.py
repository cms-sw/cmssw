import os,coral
from RecoLuminosity.LumiDB import nameDealer,dbUtil,revisionDML,lumiTime
import array

#
# Data DML API
#

#==============================
# SELECT
#==============================
def runsummary(schema,runnum,sessionflavor=''):
    '''
    select fillnum,sequence,hltkey,to_char(starttime),to_char(stoptime),egev,amodetag from cmsrunsummary where runnum=:runnum
    output: [fillnum,sequence,hltkey,l1key,starttime,stoptime]
    '''
    result=[]
    qHandle=schema.newQuery()
    t=lumiTime.lumiTime()
    try:
        qHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qHandle.addToOutputList('FILLNUM','fillnum')
        qHandle.addToOutputList('SEQUENCE','sequence')
        qHandle.addToOutputList('HLTKEY','hltkey')
        qHandle.addToOutputList('L1KEY','l1key')
        qHandle.addToOutputList('EGEV','egev')
        qHandle.addToOutputList('AMODETAG','amodetag')
        if sessionflavor=='SQLite':
            qHandle.addToOutputList('STARTTIME','starttime')
            qHandle.addToOutputList('STOPTIME','stoptime')
        else:
            qHandle.addToOutputList('to_char(STARTTIME,\''+t.coraltimefm+'\')','starttime')
            qHandle.addToOutputList('to_char(STOPTIME,\''+t.coraltimefm+'\')','stoptime')
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qResult=coral.AttributeList()
        qResult.extend('fillnum','unsigned int')
        qResult.extend('sequence','string')
        qResult.extend('hltkey','string')
        qResult.extend('l1key','string')
        qResult.extend('starttime','string')
        qResult.extend('stoptime','string')
        qResult.extend('egev','unsigned int')
        qResult.extend('amodetag','string')
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        while cursor.next():
            result.append(cursor.currentRow()['fillnum'].data())
            result.append(cursor.currentRow()['sequence'].data())
            result.append(cursor.currentRow()['hltkey'].data())
            result.append(cursor.currentRow()['l1key'].data())
            result.append(cursor.currentRow()['starttime'].data())
            result.append(cursor.currentRow()['stoptime'].data())
            result.append(cursor.currentRow()['egev'].data())
            result.append(cursor.currentRow()['amodetag'].data())
    except :
        del qHandle
        raise
    del qHandle
    return result
def luminormById(schema,dataid):
    '''
    select entry_name,amodetag,norm_1,egev_1,norm_2,egev_2 from luminorms where DATA_ID=:dataid
    result [name(0),amodetag(1),norm_1(2),egev_1(3),norm_2(4),energy_2(5) ]
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.luminormTableName())
        qHandle.addToOutputList('ENTRY_NAME','normname')
        qHandle.addToOutputList('AMODETAG','amodetag')
        qHandle.addToOutputList('NORM_1','norm_1')
        qHandle.addToOutputList('EGEV_1','energy_1')
        qHandle.addToOutputList('NORM_2','norm_2')
        qHandle.addToOutputList('EGEV_2','energy_2')        
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('normname','string')
        qResult.extend('amodetag','string')
        qResult.extend('norm_1','float')
        qResult.extend('energy_1','unsigned int')
        qResult.extend('norm_2','float')
        qResult.extend('energy_2','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:dataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            normname=cursor.currentRow()['normname'].data()
            amodetag=cursor.currentRow()['amodetag'].data()
            norm_1=cursor.currentRow()['norm_1'].data()
            energy_1=cursor.currentRow()['energy_1'].data()
            norm_2=None
            if cursor.currentRow()['norm_2'].data():
                norm_2=cursor.currentRow()['norm_2'].data()
            energy_2=None
            if cursor.currentRow()['energy_2'].data():
                energy_2=cursor.currentRow()['energy_2'].data()
            result.extend([normname,amodetag,norm_1,energy_1,norm_2,energy_2])
    except :
        del qHandle
        raise
    del qHandle
    return result

def trgRunById(schema,dataid):
    '''
    select RUNNUM,SOURCE,BITZERONAME,BITNAMECLOB from trgdata where DATA_ID=:dataid
    result [runnum(0),datasource(1),bitzeroname(2),bitnameclob(3)]
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.trgdataTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('SOURCE','source')
        qHandle.addToOutputList('BITZERONAME','bitzeroname')
        qHandle.addToOutputList('BITNAMECLOB','bitnameclob')
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned long long')
        qResult.extend('source','unsigned long long')
        qResult.extend('bitzeroname','string')
        qResult.extend('bitnameclob','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:dataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            source=cursor.currentRow()['source'].data()
            bitzeroname=cursor.currentRow()['bitzeroname'].data()
            bitnameclob=cursor.currentRow()['bitnameclob'].data()
            #print 'bitnameclob ',bitnameclob
            result.extend([runnum,source,bitzeroname,bitnameclob])
    except :
        del qHandle
        raise 
    del qHandle
    return result

def trgLSById(schema,dataid,withblobdata=False):
    '''
    result (runnum,{cmslsnum:[deadtimecount(0),bitzerocount(1),bitzeroprescale(2),deadfrac(3),prescalesblob(4),trgcountblob(5)]})
    '''
    runnum=0
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lstrgTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('DEADTIMECOUNT','deadtimecount')
        qHandle.addToOutputList('BITZEROCOUNT','bitzerocount')
        qHandle.addToOutputList('BITZEROPRESCALE','bitzeroprescale')
        qHandle.addToOutputList('DEADFRAC','deadfrac')
        if withblobdata:
            qHandle.addToOutputList('PRESCALESBLOB','prescalesblob')
            qHandle.addToOutputList('TRGCOUNTBLOB','trgcountblob')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('deadtimecount','unsigned long long')
        qResult.extend('bitzerocount','unsigned int')
        qResult.extend('bitzeroprescale','unsigned int')
        qResult.extend('deadfrac','float')
        if withblobdata:
            qResult.extend('prescalesblob','blob')
            qResult.extend('trgcountblob','blob')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            deadtimecount=cursor.currentRow()['deadtimecount'].data()
            bitzerocount=cursor.currentRow()['bitzerocount'].data()
            bitzeroprescale=cursor.currentRow()['bitzeroprescale'].data()
            deadfrac=cursor.currentRow()['deadfrac'].data()
            if not result.has_key(cmslsnum):
                result[cmslsnum]=[]
            result[cmslsnum].append(deadtimecount)
            result[cmslsnum].append(bitzerocount)
            result[cmslsnum].append(bitzeroprescale)
            result[cmslsnum].append(deadfrac)
            prescalesblob=None
            trgcountblob=None
            if withblobdata:
                prescalesblob=cursor.currentRow()['prescalesblob']
                trgcountblob=cursor.currentRow()['trgcountblob']
                result[cmslsnum].extend([prescalesblob,trgcountblob])
    except:
        del qHandle
        raise 
    del qHandle
    return (runnum,result)
def lumiRunById(schema,dataid):
    '''
    result [runnum(0),datasource(1)]
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumidataTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('SOURCE','datasource')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('datasource','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            datasource=cursor.currentRow()['datasource'].data()
            result.extend([runnum,datasource])
    except :
        del qHandle
        raise    
    del qHandle
    return result
def lumiLSById(schema,dataid,withblobdata=False):
    '''
    result (runnum,{lumilsnum,[cmslsnum(0),instlumi(1),instlumierr(2),instlumiqlty(3),beamstatus(4),beamenergy(5),numorbit(6),startorbit(7),bxindexblob(8),beam1intensity(9),beam2intensity(10)]})
    '''
    runnum=0
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumisummaryTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('LUMILSNUM','lumilsnum')
        qHandle.addToOutputList('INSTLUMI','instlumi')
        qHandle.addToOutputList('INSTLUMIERROR','instlumierr')
        qHandle.addToOutputList('INSTLUMIQUALITY','instlumiqlty')
        qHandle.addToOutputList('BEAMSTATUS','beamstatus')
        qHandle.addToOutputList('BEAMENERGY','beamenergy')
        qHandle.addToOutputList('NUMORBIT','numorbit')
        qHandle.addToOutputList('STARTORBIT','startorbit')
        if withblobdata:
            qHandle.addToOutputList('CMSBXINDEXBLOB','bxindexblob')
            qHandle.addToOutputList('BEAMINTENSITYBLOB_1','beam1intensity')
            qHandle.addToOutputList('BEAMINTENSITYBLOB_2','beam2intensity')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('lumilsnum','unsigned int')
        qResult.extend('instlumi','float')
        qResult.extend('instlumierr','float')
        qResult.extend('instlumiqlty','short')
        qResult.extend('beamstatus','string')
        qResult.extend('beamenergy','float')
        qResult.extend('numorbit','unsigned int')
        qResult.extend('startorbit','unsigned int')
        if withblobdata:
            qResult.extend('bxindexblob','blob')
            qResult.extend('beam1intensity','blob')
            qResult.extend('beam2intensity','blob')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            lumilsnum=cursor.currentRow()['lumilsnum'].data()
            instlumi=cursor.currentRow()['instlumi'].data()
            instlumierr=cursor.currentRow()['instlumierr'].data()
            instlumiqlty=cursor.currentRow()['instlumiqlty'].data()
            beamstatus=cursor.currentRow()['beamstatus'].data()
            beamenergy=cursor.currentRow()['beamenergy'].data()
            numorbit=cursor.currentRow()['numorbit'].data()
            startorbit=cursor.currentRow()['startorbit'].data()
            if not result.has_key(lumilsnum):
                result[lumilsnum]=[]
            bxindexblob=None
            beam1intensity=None
            beam2intensity=None
            if withblobdata:
                bxindexblob=cursor.currentRow()['bxindexblob'].data()
                beam1intensity=cursor.currentRow()['beam1intensity'].data()
                beam2intensity=cursor.currentRow()['beam2intensity'].data()
            result[lumilsnum].extend([cmslsnum,instlumi,instlumierr,instlumiqlty,beamstatus,beamenergy,numorbit,startorbit,bxindexblob,beam1intensity,beam2intensity])           
    except :
        del qHandle
        raise 
    del qHandle
    return (runnum,result)
def beamInfoById(schema,dataid):
    '''
    result (runnum,{lumilsnum,[cmslsnum(0),beamstatus(1),beamenergy(2),beam1intensity(3),beam2intensity(4)]})
    '''
    runnum=0
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumisummaryTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('LUMILSNUM','lumilsnum')
        qHandle.addToOutputList('BEAMSTATUS','beamstatus')
        qHandle.addToOutputList('BEAMENERGY','beamenergy')
        qHandle.addToOutputList('CMSBXINDEXBLOB','bxindexblob')
        qHandle.addToOutputList('BEAMINTENSITYBLOB_1','beam1intensity')
        qHandle.addToOutputList('BEAMINTENSITYBLOB_2','beam2intensity')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('lumilsnum','unsigned int')
        qResult.extend('beamstatus','string')
        qResult.extend('beamenergy','float')
        qResult.extend('bxindexblob','blob')
        qResult.extend('beam1intensity','blob')
        qResult.extend('beam2intensity','blob')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            lumilsnum=cursor.currentRow()['lumilsnum'].data()
            beamstatus=cursor.currentRow()['beamstatus'].data()
            beamenergy=cursor.currentRow()['beamenergy'].data()
            bxindexblob=cursor.currentRow()['bxindexblob'].data()
            beam1intensity=cursor.currentRow()['beam1intensity'].data()
            beam2intensity=cursor.currentRow()['beam2intensity'].data()
            if not result.has_key(lumilsnum):
                result[lumilsnum]=[]
            result[lumilsnum].extend([lumilsnum,beamstatus,beamenergy,bxindexblob,beam1intensity,beam2intensity])
    except :
        del qHandle
        raise
    del qHandle
    return (runnum,result)
def lumiBXById(schema,dataid):
    '''
    result {algoname,{lumilsnum:[cmslsnum,norbit,[bxlumivalue(0),bxlumierr(1),bxlumiqlty(2)]]}}
    '''
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumidetailTableName())
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('LUMILSNUM','lumilsnum')
        qHandle.addToOutputList('ALGONAME','algoname')
        qHandle.addToOutputList('BXLUMIVALUE','bxlumivalue')
        qHandle.addToOutputList('BXLUMIERROR','bxlumierr')
        qHandle.addToOutputList('BXLUMIQUALITY','bxlumiqlty')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('lumilsnum','unsigned int')
        qResult.extend('algoname','string')
        qResult.extend('bxlumivalue','blob')
        qResult.extend('bxlumierr','blob')
        qResult.extend('bxlumiqlty','blob')
        qHandle.defineOutput(qResult)
        qHandle.addToOrderList('algoname')
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            lumilsnum=cursor.currentRow()['lumilsnum'].data()
            algoname=cursor.currentRow()['algoname'].data()
            bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
            bxlumierr=cursor.currentRow()['bxlumierr'].data()
            bxlumiqlty=cursor.currentRow()['bxlumiqlty'].data()
            if not result.has_key(algoname):
                result[algoname]={}
            if not result[algoname].has_key(lumilsnum):
                result[algoname][lumilsnum]=[]
            result[algoname][lumilsnum].extend([cmslsnum,bxlumivalue,bxlumierr,bxlumiqlty])
    except :
        del qHandle
        raise RuntimeError(' dataDML.lumiBXById: '+str(e)) 
    del qHandle
    return result
def lumiBXByAlgo(schema,dataid,algoname):
    '''
    result {lumilsnum:[cmslsnum,norbit,bxlumivalue(0),bxlumierr(1),bxlumiqlty(2)]}
    '''
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumidetailTableName())
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('LUMILSNUM','lumilsnum')
        qHandle.addToOutputList('BXLUMIVALUE','bxlumivalue')
        qHandle.addToOutputList('BXLUMIERROR','bxlumierr')
        qHandle.addToOutputList('BXLUMIQUALITY','bxlumiqlty')
        qConditionStr='ALGONAME=:algoname and DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition.extend('algoname','string')
        qCondition['dataid'].setData(dataid)
        qCondition['algoname'].setData(algoname)
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('lumilsnum','unsigned int')
        qResult.extend('bxlumivalue','blob')
        qResult.extend('bxlumierr','blob')
        qResult.extend('bxlumiqlty','blob')
        qHandle.defineOutput(qResult)
        qHandle.addToOrderList('algoname')
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            lumilsnum=cursor.currentRow()['lumilsnum'].data()
            bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
            bxlumierr=cursor.currentRow()['bxlumierr'].data()
            bxlumiqlty=cursor.currentRow()['bxlumiqlty'].data()
            if not result.has_key(lumilsnum):
                result[lumilsnum]=[]
            result[lumilsnum].extend([cmslsnum,bxlumivalue,bxlumierr,bxlumiqlty])
    except :
        del qHandle
        raise 
    del qHandle
    return result
def hltRunById(schema,dataid):
    '''
    result [runnum(0),datasource(1),npath(2),pathnameclob(3)]
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.hltdataTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('SOURCE','datasource')
        qHandle.addToOutputList('NPATH','npath')
        qHandle.addToOutputList('PATHNAMECLOB','pathnameclob')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('datasource','string')
        qResult.extend('npath','unsigned int')
        qResult.extend('pathnameclob','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            datasource=cursor.currentRow()['datasource'].data()
            npath=cursor.currentRow()['npath'].data()
            pathnameclob=cursor.currentRow()['pathnameclob'].data()
            result.extend([runnum,datasource,npath,pathnameclob])
    except :
        del qHandle
        raise 
    del qHandle
    return result
def hltLSById(schema,dataid):
    '''
    result (runnum, {cmslsnum:[prescaleblob,hltcountblob,hltacceptblob]} 
    '''
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lshltTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('PRESCALEBLOB','prescaleblob')
        qHandle.addToOutputList('HLTCOUNTBLOB','hltcountblob')
        qHandle.addToOutputList('HLTACCEPTBLOB','hltacceptblob')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('prescaleblob','blob')
        qResult.extend('hltcountblob','blob')
        qResult.extend('hltacceptblob','blob')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            prescaleblob=cursor.currentRow()['prescaleblob'].data()
            hltcountblob=cursor.currentRow()['hltcountblob'].data()
            hltacceptblob=cursor.currentRow()['hltacceptblob'].data()
            if not result.has_key(cmslsnum):
                result[cmslsnum]=[]
            result[cmslsnum].extend([prescaleblob,hltcountblob,hltacceptblob])
    except :
        del qHandle
        raise
    del qHandle
    return (runnum,result)
def guessDataIdByRun(schema,runnum):
    '''
    get dataids by runnumber, if there are duplicates, pick max(dataid).Bypass full version lookups
    result (lumidataid(0),trgdataid(1),hltdataid(2)) 
    '''
    lumiids=[]
    trgids=[]
    hltids=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumidataTableName(),'l')
        qHandle.addToTableList(nameDealer.trgdataTableName(),'t')
        qHandle.addToTableList(nameDealer.hltdataTableName(),'h')
        qHandle.addToOutputList('l.DATA_ID','lumidataid')
        qHandle.addToOutputList('t.DATA_ID','trgdataid')
        qHandle.addToOutputList('h.DATA_ID','hltdataid')
        qConditionStr='l.RUNNUM=t.RUNNUM and t.RUNNUM=h.RUNNUM and l.RUNNUM=:runnum '
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(runnum)
        qResult=coral.AttributeList()
        qResult.extend('lumidataid','unsigned long long')
        qResult.extend('trgdataid','unsigned long long')
        qResult.extend('hltdataid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            lumidataid=cursor.currentRow()['lumidataid'].data()
            trgdataid=cursor.currentRow()['trgdataid'].data()
            hltdataid=cursor.currentRow()['hltdataid'].data()
            lumiids.append(lumidataid)
            trgids.append(trgdataid)
            hltids.append(hltdataid)
    except :
        del qHandle
        raise 
    del qHandle
    return (max(lumiids),max(trgids),max(hltids))

def guessnormIdByContext(schema,amodetag,egev1):
    '''
    get norm dataids by amodetag, egev if there are duplicates, pick max(dataid).Bypass full version lookups
    select data_id from luminorm where amodetag=:amodetag and egev_1=:egev1   
    '''
    luminormids=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.luminormTableName() )
        qHandle.addToOutputList('DATA_ID','normdataid')
        qConditionStr='AMODETAG=:amodetag AND EGEV_1=:egev1'
        qCondition=coral.AttributeList()
        qCondition.extend('amodetag','string')
        qCondition.extend('egev1','unsigned int')
        qCondition['amodetag'].setData(amodetag)
        qCondition['egev1'].setData(egev1)
        qResult=coral.AttributeList()
        qResult.extend('normdataid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            normdataid=cursor.currentRow()['normdataid'].data()
            luminormids.append(normdataid)
    except :
        del qHandle
        raise
    del qHandle
    if len(luminormids) !=0:return max(luminormids)
    return None

def guessnormIdByName(schema,normname):
    '''
    get norm dataids by name, if there are duplicates, pick max(dataid).Bypass full version lookups
    select luminorm.data_id from luminorm where name=:normname
    result luminormdataid
    '''   
    luminormids=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.entryTableName(nameDealer.luminormTableName()) )
        qHandle.addToTableList( nameDealer.luminormTableName() )
        qHandle.addToOutputList('DATA_ID','normdataid')
        qConditionStr='ENTRY_NAME=:normname '
        qCondition=coral.AttributeList()
        qCondition.extend('normname','string')
        qCondition['normname'].setData(normname)
        qResult=coral.AttributeList()
        qResult.extend('normdataid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            normdataid=cursor.currentRow()['normdataid'].data()
            luminormids.append(normdataid)
    except :
        del qHandle
        raise
    del qHandle
    if len(luminormids) !=0:return max(luminormids)
    return None

########
########
def dataentryIdByRun(schema,runnum,branchfilter):
    '''
    select el.entry_id,et.entry_id,eh.entry_id,el.revision_id,et.revision_id,eh.revision_id from lumidataentiries el,trgdataentries et,hltdataentries eh where el.name=et.name and et.name=eh.name and el.name=:entryname;
    check on entryrev
   
    return [lumientryid,trgentryid,hltentryid]
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.entryTableName( lumidataTableName() ))
        qHandle.addToTableList(nameDealer.entryTableName( trgdataTableName() ))
        qHandle.addToTableList(nameDealer.entryTableName( hltdataTableName() ))
        qHandle.addToOutputList(lumidataTableName()+'.ENTRY_ID','lumientryid')
        qHandle.addToOutputList(trgdataTableName()+'.ENTRY_ID','trgentryid')
        qHandle.addToOutputList(hltdataTableName()+'.ENTRY_ID','hltentryid')
        qConditionStr=lumidataTableName()+'.NAME='+trgdataTableName()+'.NAME AND '+trgdataTableName()+'.NAME='+hltdataTableName()+'.NAME AND '+lumidataTableName()+'.NAME=:runnumstr'
        qCondition=coral.AttributeList()
        qCondition.extend('runnumstr','string')
        qCondition['runnumstr'].setData(str(runnum))
        qResult=coral.AttributeList()
        qResult.extend('lumientryid','unsigned long long')
        qResult.extend('trgentryid','unsigned long long')
        qResult.extend('hltentryid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            lumientryid=cursor.currentRow()['lumientryid'].data()
            trgentryid=cursor.currentRow()['trgentryid'].data()
            hltentryid=cursor.currentRow()['hltentryid'].data()
            if lumientryid in branchfilter and trgentryid in branchfilter and hltentryid in branchfilter:
                result.extend([lumientryid,trgentryid,hltentryid])
    except:
        del qHandle
        raise 
    del qHandle
    return result

def latestdataIdByEntry(schema,entryid,datatype,branchfilter):
    '''
    select l.data_id,rl.revision_id from lumidatatable l,lumirevisions rl where  l.data_id=rl.data_id and l.entry_id=:entryid
    check revision_id is in branch
    '''
    dataids=[]
    datatablename=''
    revmaptablename=''
    if datatype=='lumi':
        datatablename=nameDealer.lumidataTableName()
    elif datatype=='trg':
        datatablename=nameDealer.trgdataTableName()
    elif dataytpe=='hlt':
        tablename=nameDealer.hltdataTableName()
    else:
        raise RunTimeError('datatype '+datatype+' is not supported')
    revmaptablename=nameDealer.revmapTableName(datatablename)
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(revmaptablename)
        qHandle.addToTableList(datatablename)
        qHandle.addToOutputList('l.DATA_ID','dataid')
        qHandle.addToOutputList(revmaptablename+'.REVISION_ID','revisionid')
        qConditionStr=datatablename+'.DATA_ID='+revmaptablename+'.DATA_ID AND '+datatablename+'.ENTRY_ID=:entryid'
        qCondition=coral.AttributeList()
        qCondition.extend('entryid','unsigned long long')
        qResult=coral.AttributeList()
        qResult.extend('dataid','unsigned long long')
        qResult.extend('revisionid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            dataid=cursor.currentRow()['dataid'].data()
            revisionid=cursor.currentRow()['revisionid'].data()
            if revisionid in branchfilter:
                dataids.append(dataid)
    except:
        del qHandle
        raise
    del qHandle
    if len(dataids)!=0:return max(dataids)
    return None

#=======================================================
#   INSERT requires in update transaction
#=======================================================
def addNormToBranch(schema,normname,amodetag,norm1,egev1,optionalnormdata,branchinfo):
    '''
    input:
       optionalnormdata {'norm2':norm2,'egev2':egev2}
    output:
       (revision_id,entry_id,data_id)
    '''
    norm2=None
    if optionalnormdata.has_key('norm2'):
        norm2=optionalnormdata['norm2']
    egev2=None
    if optionalnormdata.has_key('egev2'):
        egev2=optionalnormdata['egev2']
    try:
        entry_id=revisionDML.entryInBranch(schema,nameDealer.luminormTableName(),normname,branchinfo[1])
        if entry_id is None:
            (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.luminormTableName())
            entryinfo=(revision_id,entry_id,normname,data_id)
            revisionDML.addEntry(schema,nameDealer.luminormTableName(),entryinfo,branchinfo)
        else:
            (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.luminormTableName() )
            revisionDML.addRevision(schema,nameDealer.luminormTableName(),(revision_id,data_id),branchinfo)
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','AMODETAG':'string','NORM_1':'float','EGEV_1':'unsigned int','NORM_2':'float','EGEV_2':'unsigned int'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':normname,'AMODETAG':amodetag,'NORM_1':norm1,'EGEV_1':egev1,'NORM_2':norm2,'EGEV_2':egev2}
        db=dbUtil.dbUtil(schema)
        db.insertOneRow(nameDealer.luminormTableName(),tabrowDefDict,tabrowValueDict)
        return (revision_id,entry_id,data_id)
    except :
        raise
def addLumiRunDataToBranch(schema,runnumber,lumirundata,branchinfo):
    '''
    input:
          lumirundata [datasource]
          branchinfo (branch_id,branch_name)
    output:
          (revision_id,entry_id,data_id)
    '''
    try:
         datasource=lumirundata[0]
         entry_id=revisionDML.entryInBranch(schema,nameDealer.lumidataTableName(),str(runnumber),branchinfo[1])
         if entry_id is None:
             (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.lumidataTableName())
             entryinfo=(revision_id,entry_id,str(runnumber),data_id)
             revisionDML.addEntry(schema,nameDealer.lumidataTableName(),entryinfo,branchinfo)
         else:
             (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.lumidataTableName() )
             revisionDML.addRevision(schema,nameDealer.lumidataTableName(),(revision_id,data_id),branchinfo)
         tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','RUNNUM':'unsigned int','SOURCE':'string'}
         tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':str(runnumber),'RUNNUM':int(runnumber),'SOURCE':datasource}
         db=dbUtil.dbUtil(schema)
         db.insertOneRow(nameDealer.lumidataTableName(),tabrowDefDict,tabrowValueDict)
         return (revision_id,entry_id,data_id)
    except :
        raise
def addTrgRunDataToBranch(schema,runnumber,trgrundata,branchinfo):
    '''
    input:
       trgrundata [datasource(0),bitzeroname(1),bitnameclob(2)]
       bitnames clob, bitnames separated by ','
    output:
       (revision_id,entry_id,data_id)
    '''
    try:   #fixme: need to consider revision only case
        datasource=trgrundata[0]
        bitzeroname=trgrundata[1]
        bitnames=trgrundata[2]
        entry_id=revisionDML.entryInBranch(schema,nameDealer.trgdataTableName(),str(runnumber),branchinfo[1])
        if entry_id is None:
            (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.trgdataTableName())
            entryinfo=(revision_id,entry_id,str(runnumber),data_id)
            revisionDML.addEntry(schema,nameDealer.trgdataTableName(),entryinfo,branchinfo)
        else:
            (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.trgdataTableName() )
            revisionDML.addRevision(schema,nameDealer.trgdataTableName(),(revision_id,data_id),branchinfo)
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','SOURCE':'string','RUNNUM':'unsigned int','BITZERONAME':'string','BITNAMECLOB':'string'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':str(runnumber),'SOURCE':datasource,'RUNNUM':int(runnumber),'BITZERONAME':bitzeroname,'BITNAMECLOB':bitnames}
        db=dbUtil.dbUtil(schema)
        db.insertOneRow(nameDealer.trgdataTableName(),tabrowDefDict,tabrowValueDict)
        return (revision_id,entry_id,data_id)
    except :
        raise
def addHLTRunDataToBranch(schema,runnumber,hltrundata,branchinfo):
    '''
    input:
        hltrundata [pathnameclob(0),datasource(1)]
    output:
        (revision_id,entry_id,data_id)
    '''
    try:
         pathnames=hltrundata[0]
         datasource=hltrundata[1]
         npath=len(pathnames.split(','))
         entry_id=revisionDML.entryInBranch(schema,nameDealer.lumidataTableName(),str(runnumber),branchinfo[1])
         if entry_id is None:
             (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.hltdataTableName())
             entryinfo=(revision_id,entry_id,str(runnumber),data_id)
             revisionDML.addEntry(schema,nameDealer.hltdataTableName(),entryinfo,branchinfo)
         else:
             (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.hltdataTableName() )
             revisionDML.addRevision(schema,nameDealer.hltdataTableName(),(revision_id,data_id),branchinfo)
         tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','RUNNUM':'unsigned int','SOURCE':'string','NPATH':'unsigned int','PATHNAMECLOB':'string'}
         tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':str(runnumber),'RUNNUM':int(runnumber),'SOURCE':datasource,'NPATH':npath,'PATHNAMECLOB':pathnames}
         db=dbUtil.dbUtil(schema)
         db.insertOneRow(nameDealer.hltdataTableName(),tabrowDefDict,tabrowValueDict)
         return (revision_id,entry_id,data_id)
    except :
        raise 

def insertRunSummaryData(schema,runnumber,runsummarydata):
    '''
    input:
        runsummarydata [hltkey,l1key,fillnum,sequence,starttime,stoptime,amodetag,egev]
    output:
    
    '''
    try:
        tabrowDefDict={'RUNNUM':'unsigned int','HLTKEY':'string','L1KEY':'string','SEQUENCE':'string','FILLNUM':'unsigned int','STARTTIME':'time stamp','STOPTIME':'time stamp','AMODETAG':'string','EGEV':'unsigned int'}
        tabrowValueDict={'RUNNUM':int(runnumber),'HLTKEY':runsummarydata[0],'L1KEY':runsummarydata[1],'SEQUENCE':runsummarydata[2],'FILLNUM':runsummarydata[3],'STARTTIME':runsummarydata[4],'STOPTIME':runsummarydata[5],'AMODETAG':runsummarydata[6],'EGEV':runsummarydata[7]}
        db=dbUtil.dbUtil(schema)
        db.insertOneRow(nameDealer.cmsrunsummaryTableName(),tabrowDefDict,tabrowValueDict)
    except :
        raise   
def insertTrgHltMap(schema,hltkey,trghltmap):
    '''
    input:
        trghltmap {hltpath:(l1seed,hltpathid)}
    output:
    '''
    try:
        nrows=0
        bulkvalues=[]   
        trghltDefDict=[('HLTKEY','string'),('HLTPATHNAME','string'),('L1SEED','string'),('HLTPATHID','unsigned int')]
        for hltpath,(l1seed,hltpathid) in trghltmap.items():
            bulkvalues.append([('HLTKEY',hltkey),('HLTPATHNAME',hltpath),('L1SEED',l1seed),('HLTPATHID',hltpathid)])
        db=dbUtil.dbUtil(schema)
        db.bulkInsert(nameDealer.trghltMapTableName(),trghltDefDict,bulkvalues)
        nrows=len(bulkvalues)
        return nrows
    except :
        raise
def insertTrgLSData(schema,runnumber,data_id,trglsdata):
    '''
    insert trg per-LS data for given run and data_id, this operation can be split in transaction chuncks 
    input:
        trglsdata {cmslsnum:[deadtime,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}
    result nrows inserted
    if nrows==0, then this insertion failed
    '''
    try:
        nrows=0
        bulkvalues=[]   
        lstrgDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('CMSLSNUM','unsigned int'),('DEADTIMECOUNT','unsigned long long'),('BITZEROCOUNT','unsigned int'),('BITZEROPRESCALE','unsigned int'),('PRESCALEBLOB','blob'),('TRGCOUNTBLOB','blob')]
        for cmslsnum,perlstrg in trglsdata.items():
            deadtimecount=perlstrg[0]           
            bitzerocount=perlstrg[1]
            bitzeroprescale=perlstrg[2]
            trgcountblob=perlstrg[3]
            trgprescaleblob=perlstrg[4]
            bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('CMSLSNUM',cmslsnum),('DEADTIMECOUNT',deadtimecount),('BITZEROCOUNT',bitzerocount),('BITZEROPRESCALE',bitzeroprescale),('PRESCALEBLOB',trgprescaleblob),('TRGCOUNTBLOB',trgcountblob)])
        db=dbUtil.dbUtil(schema)
        db.bulkInsert(nameDealer.lstrgTableName(),lstrgDefDict,bulkvalues)
        nrows=len(bulkvalues)
        return nrows
    except :
        raise 
def insertHltLSData(schema,runnumber,data_id,hltlsdata):
    '''
    input:
         hltlsdata {cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}
    '''
    try:
        nrow=0
        bulkvalues=[]
        lshltDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('CMSLSNUM','unsigned int'),('PRESCALEBLOB','blob'),('HLTCOUNTBLOB','blob'),('HLTACCEPTBLOB','blob')]
        for cmslsnum,perlshlt in hltlsdata.items():
            inputcountblob=perlshlt[0]
            acceptcountblob=perlshlt[1]
            prescaleblob=perlshlt[2]
            bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('CMSLSNUM',cmslsnum),('PRESCALEBLOB',prescaleblob),('HLTCOUNTBLOB',inputcountblob),('HLTACCEPTBLOB',acceptcountblob)])
        db=dbUtil.dbUtil(schema)
        db.bulkInsert(nameDealer.lshltTableName(),lshltDefDict,bulkvalues)
        return len(bulkvalues)
    except Exception,e :
        raise RuntimeError(' dataDML.addHltLSData: '+str(e))
    
def insertLumiLSSummary(schema,runnumber,data_id,lumilsdata):
    '''
    input:
          lumilsdata {lumilsnum:[cmslsnum,cmsalive,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexindexblob,beam1intensity,beam2intensity]}
    output:
          nrows
    '''
    try:
        nrow=0
        bulkvalues=[]
        lslumiDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('LUMILSNUM','unsigned int'),('CMSLSNUM','unsigned int'),('CMSALIVE','short'),('INSTLUMI','float'),('INSTLUMIERROR','float'),('INSTLUMIQUALITY','short'),('BEAMSTATUS','string'),('BEAMENERGY','float'),('NUMORBIT','unsigned int'),('STARTORBIT','unsigned int'),('CMSBXINDEXBLOB','blob'),('BEAMINTENSITYBLOB_1','blob'),('BEAMINTENSITYBLOB_2','blob')]
        for lumilsnum,perlslumi in lumilsdata.items():
            cmslsnum=perlslumi[0]
            cmsalive=perlslumi[1]
            instlumi=perlslumi[2]
            instlumierror=perlslumi[3]
            instlumiquality=perlslumi[4]
            beamstatus=perlslumi[5]
            beamenergy=perlslumi[6]
            numorbit=perlslumi[7]
            startorbit=perlslumi[8]
            cmsbxindexindexblob=perlslumi[9]
            beam1intensity=perlslumi[10]
            beam2intensity=perlslumi[11]
            bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('LUMILSNUM',lumilsnum),('CMSLSNUM',cmslsnum),('CMSALIVE',cmsalive),('INSTLUMI',instlumi),('INSTLUMIERROR',instlumierror),('INSTLUMIQUALITY',instlumiquality),('BEAMSTATUS',beamstatus),('CMSBXINDEXBLOB',beam1intensity),('BEAMINTENSITYBLOB_1',beam1intensity),('BEAMINTENSITYBLOB_2',beam2intensity),('NUMORBIT',numorbit),('STARTORBIT',startorbit)])
        db=dbUtil.dbUtil(schema)
        db.bulkInsert(nameDealer.lumisummaryTableName(),lslumiDefDict,bulkvalues)
        return len(bulkvalues)
    except :
        raise

def insertLumiLSDetail(schema,runnumber,data_id,lumibxdata):
    '''
    input:
          lumibxdata [(algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]}),(algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]}),(algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]})]
    output:
          nrows
    '''
    try:
        nrow=0
        bulkvalues=[]
        lslumiDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('LUMILSNUM','unsigned int'),('CMSLSNUM','unsigned int'),('ALGONAME','string'),('BXLUMIVALUE','blob'),('BXLUMIERROR','blob'),('BXLUMIQUALITY','blob')]
        for (algoname,peralgobxdata) in lumibxdata:
            for lumilsnum,bxdata in peralgobxdata.items():
                cmslsnum=bxdata[0]
                bxlumivalue=bxdata[1]
                bxlumierror=bxdata[2]
                bxlumiquality=bxdata[3]
                bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('LUMILSNUM',lumilsnum),('CMSLSNUM',cmslsnum),('ALGONAME',algoname),('BXLUMIVALUE',bxlumivalue),('BXLUMIERROR',bxlumierror),('BXLUMIQUALITY',bxlumiquality)])
        db=dbUtil.dbUtil(schema)
        db.bulkInsert(nameDealer.lumidetailTableName(),lslumiDefDict,bulkvalues)
        return len(bulkvalues)
    except:
        raise 
    
def completeOldLumiData(schema,runnumber,lsdata,data_id):
    '''
    input:
    lsdata [[lumisummary_id,lumilsnum,cmslsnum]]
    '''
    try:
        #update in lumisummary table
        #print 'insert in lumisummary table'
        setClause='DATA_ID=:data_id'
        updateCondition='RUNNUM=:runnum AND DATA_ID is NULL'
        updateData=coral.AttributeList()
        updateData.extend('data_id','unsigned long long')
        updateData.extend('runnum','unsigned int')
        updateData['data_id'].setData(data_id)
        updateData['runnum'].setData(int(runnumber))
        db=dbUtil.dbUtil(schema)
        db.singleUpdate(nameDealer.lumisummaryTableName(),setClause,updateCondition,updateData)
        #updates in lumidetail table
        updateAction='DATA_ID=:data_id,RUNNUM=:runnum,CMSLSNUM=:cmslsnum,LUMILSNUM=:lumilsnum'
        updateCondition='LUMISUMMARY_ID=:lumisummary_id'
        bindvarDef=[]
        bindvarDef.append(('data_id','unsigned long long'))
        bindvarDef.append(('runnum','unsigned int'))
        bindvarDef.append(('cmslsnum','unsigned int'))
        bindvarDef.append(('lumilsnum','unsigned int'))        
        inputData=[]
        for [lumisummary_id,lumilsnum,cmslsnum] in lsdata:
            inputData.append([('data_id',data_id),('runnum',int(runnumber)),('cmslsnum',cmslsnum),('lumilsnum',lumilsnum)])
        db.updateRows(nameDealer.lumidetailTableName(),updateAction,updateCondition,bindvarDef,inputData)
    except:
        raise
    
#=======================================================
#   DELETE
#=======================================================


#=======================================================
#   Unit Test
#=======================================================
if __name__ == "__main__":
    import sessionManager
    import lumidbDDL,revisionDML,generateDummyData
    myconstr='sqlite_file:test2.db'
    svc=sessionManager.sessionManager(myconstr,debugON=False)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    schema=session.nominalSchema()
    session.transaction().start(False)
    tables=lumidbDDL.createTables(schema)
    try:
        lumidbDDL.createUniqueConstraints(schema)
        trunkinfo=revisionDML.createBranch(schema,'TRUNK',None,comment='main')
        #print trunkinfo
        datainfo=revisionDML.createBranch(schema,'DATA','TRUNK',comment='hold data')
        #print datainfo
        norminfo=revisionDML.createBranch(schema,'NORM','TRUNK',comment='hold normalization factor')
        #print norminfo
    except:
        print 'branch already exists, do nothing'
    (normbranchid,normbranchparent)=revisionDML.branchInfoByName(schema,'NORM')
    normbranchinfo=(normbranchid,'NORM')
    addNormToBranch(schema,'pp7TeV','PROTPHYS',6370.0,3500,{},normbranchinfo)
    addNormToBranch(schema,'hi7TeV','HIPHYS',2.38,3500,{},normbranchinfo)
    (branchid,branchparent)=revisionDML.branchInfoByName(schema,'DATA')
    branchinfo=(branchid,'DATA')
    for runnum in [1200,1211,1222,1233,1345]:
        runsummarydata=generateDummyData.runsummary(schema,'PROTPHYS',3500)
        insertRunSummaryData(schema,runnum,runsummarydata)
        hlttrgmap=generateDummyData.hlttrgmap(schema)
        insertTrgHltMap(schema,hlttrgmap[0],hlttrgmap[1])
        
        lumidummydata=generateDummyData.lumiSummary(schema,20)
        lumirundata=[lumidummydata[0]]
        lumilsdata=lumidummydata[1]
        (lumirevid,lumientryid,lumidataid)=addLumiRunDataToBranch(schema,runnum,lumirundata,branchinfo)
        insertLumiLSSummary(schema,runnum,lumidataid,lumilsdata)
        lumibxdata=generateDummyData.lumiDetail(schema,20)
        insertLumiLSDetail(schema,runnum,lumidataid,lumibxdata)      
        trgdata=generateDummyData.trg(schema,20)        
        trgrundata=[trgdata[0],trgdata[1],trgdata[2]]
        trglsdata=trgdata[3]
        (trgrevid,trgentryid,trgdataid)=addTrgRunDataToBranch(schema,runnum,trgrundata,branchinfo)
        insertTrgLSData(schema,runnum,trgdataid,trglsdata)        
        hltdata=generateDummyData.hlt(schema,20)
        hltrundata=[hltdata[0],hltdata[1]]
        hltlsdata=hltdata[2]
        (hltrevid,hltentryid,hltdataid)=addHLTRunDataToBranch(schema,runnum,hltrundata,branchinfo)
        insertHltLSData(schema,runnum,hltdataid,hltlsdata)
    session.transaction().commit()
    print 'test reading'
    session.transaction().start(True)
    print '===inspecting NORM by name==='
    normrevlist=revisionDML.revisionsInBranchName(schema,'NORM')
    luminormentry_id=revisionDML.entryInBranch(schema,nameDealer.luminormTableName(),'pp7TeV','NORM')
    latestNorms=revisionDML.latestDataRevisionOfEntry(schema,nameDealer.luminormTableName(),luminormentry_id,normrevlist)
    print 'latest norm data_id for pp7TeV ',latestNorms
    
    print '===inspecting DATA branch==='
    print revisionDML.branchType(schema,'DATA')
    revlist=revisionDML.revisionsInBranchName(schema,'DATA')
    print revlist
    lumientry_id=revisionDML.entryInBranch(schema,nameDealer.lumidataTableName(),'1211','DATA')
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,nameDealer.lumidataTableName(),lumientry_id,revlist)
    print 'latest lumi data_id for run 1211 ',latestrevision
    lumientry_id=revisionDML.entryInBranch(schema,nameDealer.lumidataTableName(),'1222','DATA')
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,nameDealer.lumidataTableName(),lumientry_id,revlist)
    print 'latest lumi data_id for run 1222 ',latestrevision
    trgentry_id=revisionDML.entryInBranch(schema,nameDealer.trgdataTableName(),'1222','DATA')
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,nameDealer.trgdataTableName(),trgentry_id,revlist)
    print 'latest trg data_id for run 1222 ',latestrevision
    session.transaction().commit()
    print 'tagging data so far as data_orig'
    session.transaction().start(False)
    (revisionid,parentid,parentname)=revisionDML.createBranch(schema,'data_orig','DATA',comment='tag of 2010data')
    session.transaction().commit()
    session.transaction().start(True)
    print revisionDML.branchType(schema,'data_orig')
    revlist=revisionDML.revisionsInTag(schema,revisionid,branchinfo[0])
    print revlist
    session.transaction().commit()
    session.transaction().start(False)
    for runnum in [1200,1222]:
        print 'revising lumidata for run ',runnum
        lumidummydata=generateDummyData.lumiSummary(schema,20)
        lumirundata=[lumidummydata[0]]
        lumilsdata=lumidummydata[1]
        (lumirevid,lumientryid,lumidataid)=addLumiRunDataToBranch(schema,runnum,lumirundata,branchinfo)
        insertLumiLSSummary(schema,runnum,lumidataid,lumilsdata)
        lumibxdata=generateDummyData.lumiDetail(schema,20)
        insertLumiLSDetail(schema,runnum,lumidataid,lumibxdata)            
    revlist=revisionDML.revisionsInTag(schema,revisionid,branchinfo[0])
    print 'revisions in branch DATA',revisionDML.revisionsInBranch(schema,branchinfo[0])
    session.transaction().commit()
    print 'revisions in tag data_orig ',revlist
    
    print '===test reading==='
    session.transaction().start(True)
    print 'guess norm by name'
    normid1=guessnormIdByName(schema,'pp7TeV')
    print 'normid1 ',normid1
    normid2=guessnormIdByContext(schema,'PROTPHYS',3500)
    print 'guess norm of PROTPHYS 3500'
    print 'normid2 ',normid2
    normid=normid2
    (lumidataid,trgdataid,hltdataid)=guessDataIdByRun(schema,1200)
    print 'normid,lumiid,trgid,hltid ',normid,lumidataid,trgdataid,hltdataid
    print 'lumi norm'
    print luminormById(schema,normid)
    print 'runinfo '
    print runsummary(schema,runnum,session.properties().flavorName())
    print 'lumirun '
    print lumiRunById(schema,lumidataid)
    print 'lumisummary'
    print lumiLSById(schema,lumidataid)
    print 'beam info'
    print beamInfoById(schema,lumidataid)
    print 'lumibx by algo OCC1'
    print lumiBXByAlgo(schema,lumidataid,'OCC1')
    print 'trg run'
    print trgRunById(schema,trgdataid)
    print 'trg ls'
    print trgLSById(schema,trgdataid)
    print 'hlt run'
    print hltRunById(schema,hltdataid)
    print 'hlt ls'
    print hltLSById(schema,hltdataid)
    session.transaction().commit()
    del session

import os,coral
from RecoLuminosity.LumiDB import nameDealer,dbUtil,revisionDML
import array

#
# Data DML API
#

#==============================
# SELECT
#==============================
def luminormById(schema,dataid):
    '''
    select name,defaultnorm,norm_1,energy_1,norm_2,energy_2 from luminorms where DATA_ID=:dataid
    result [name(0),defaultnorm(1),norm_1(2),energy_1(3),norm_2(4),energy_2(5) ]
    '''
    result=[]
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.luminormTablename())
        qHandle.addToOutputList('NAME','normname')
        qHandle.addToOutputList('DEFAULTNORM','defaultnorm')
        qHandle.addToOutputList('NORM_1','norm_1')
        qHandle.addToOutputList('ENERGY_1','energy_1')
        qHandle.addToOutputList('NORM_2','norm_2')
        qHandle.addToOutputList('ENERGY_2','energy_2')        
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('normname','string')
        qResult.extend('defaultnorm','float')
        qResult.extend('norm_1','float')
        qResult.extend('energy_1','float')
        qResult.extend('norm_2','float')
        qResult.extend('energy_2','float')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:dataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            normname=cursor.currentRow()['normname'].data()
            defaultnorm=cursor.currentRow()['defaultnorm'].data()
            norm_1=None
            if cursor.currentRow()['norm_1'].data():
                norm_1=cursor.currentRow()['norm_1'].data()
            energy_1=None
            if cursor.currentRow()['energy_1'].data():
                energy_1=cursor.currentRow()['energy_1'].data()
            norm_2=None
            if cursor.currentRow()['norm_2'].data():
                norm_2=cursor.currentRow()['norm_2'].data()
            energy_2=None
            if cursor.currentRow()['energy_2'].data():
                energy_2=cursor.currentRow()['energy_2'].data()
            result.extend([normname,defaultnorm,norm_1,energy_1,norm_2,energy_2])
        del qHandle
    except Exception,e :
        raise RuntimeError(' dataDML.luminormById: '+str(e))
    return result

def trgRunById(schema,dataid):
    '''
    select RUNNUM,SOURCE,BITZERONAME,BITNAMECLOB from trgdata where DATA_ID=:dataid
    result [runnum(0),datasource(1),bitzeroname(2),bitnameclob(3)]
    '''
    result=[]
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.trgdataTablename())
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
        qResult.extend('bitzeroclob','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:dataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            source=cursor.currentRow()['source'].data()
            bitzeroname=cursor.currentRow()['bitzeroname'].data()
            bitnameclob=cursor.currentRow()['bitzeroclob'].data()
            result.extend([runnum,source,bitzeroname,bitnameclob])
        del qHandle
    except Exception,e :
        raise RuntimeError(' dataDML.trgRunById: '+str(e))
    return result

def trgLSById(schema,dataid,withblobdata=False):
    '''
    result (runnum,{cmslsnum:[deadtimecount(0),bitzerocount(1),bitzeroprescale(2),deadfrac(3),prescalesblob(4),trgcountblob(5)]})
    '''
    runnum=0
    result={}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.lstrgTablename())
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
        del qHandle
        return (runnum,result)
    except Exception,e :
        raise RuntimeError(' dataDML.trgLSById: '+str(e))

def lumiRunById(schema,dataid):
    '''
    result [runnum(0),datasource(1)]
    '''
    result=[]
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.lumidataTablename())
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
        del qHandle
        return result
    except Exception,e :
        raise RuntimeError(' dataDML.lumiRunById: '+str(e))      

def lumiLSById(schema,dataid,withblobdata=False):
    '''
    result (runnum,{lumilsnum,[cmslsnum(0),instlumi(1),instlumierr(2),instlumiqlty(3),beamstatus(4),beamenergy(5),numorbit(6),startorbit(7),bxindexblob(8),beam1intensity(9),beam2intensity(10)]})
    '''
    runnum=0
    result={}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.lumisummaryTablename())
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
        del qHandle
        return (runnum,result)
    except Exception,e :
        raise RuntimeError(' dataDML.lumiLSById: '+str(e))  
def beamInfoById(schema,dataid):
    '''
    result (runnum,{lumilsnum,[cmslsnum(0),beamstatus(1),beamenergy(2),beam1intensity(3),beam2intensity(4)]})
    '''
    runnum=0
    result={}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.lumisummaryTablename())
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
            lumilsnum==cursor.currentRow()['lumilsnum'].data()
            beamstatus=cursor.currentRow()['beamstatus'].data()
            beamenergy=cursor.currentRow()['beamenergy'].data()
            bxindexblob=cursor.currentRow()['bxindexblob'].data()
            beam1intensity=cursor.currentRow()['beam1intensity'].data()
            beam2intensity=cursor.currentRow()['beam2intensity'].data()
            if not result.has_key(lumilsnum):
                result[lumilsnum].extend([lumilsnum,beamstatus,beamenergy,bxindexblob,beam1intensity,beam2intensity])
        del qHandle
        return (runnum,result)
    except Exception,e :
        raise RuntimeError(' dataDML.beamInfoById: '+str(e))      

def lumiBXById(schema,dataid):
    '''
    result {algoname,{lumilsnum:[cmslsnum,norbit,[bxlumivalue(0),bxlumierr(1),bxlumiqlty(2)]]}}
    '''
    result={}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.lumidetailTablename())
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
        del qHandle
        return result
    except Exception,e :
        raise RuntimeError(' dataDML.lumiBXById: '+str(e)) 

def lumiBXByAlgo(schema,dataid,algoname):
    '''
    result {lumilsnum:[cmslsnum,norbit,bxlumivalue(0),bxlumierr(1),bxlumiqlty(2)]}
    '''
    result={}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.lumidetailTablename())
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
        del qHandle
        return result
    except Exception,e :
        raise RuntimeError(' dataDML.lumiBXById: '+str(e)) 

def hltRunById(schema,dataid):
    '''
    result [runnum(0),datasource(1),npath(2),pathnameclob(3)]
    '''
    result=[]
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.hltdataTablename())
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
        del qHandle
        return result
    except Exception,e:
        raise RuntimeError(' dataDML.hltRunById: '+str(e))
    
def hltLSById(schema,dataid):
    '''
    result (runnum, {cmslsnum:[prescaleblob,hltcountblob,hltacceptblob]}
    '''
    result={}
    try:
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            prescaleblob=cursor.currentRow()['prescaleblob'].data()
            hltcountblob=cursor.currentRow()['hltcountblob'].data()
            hltacceptblob=cursor.currentRow()['hltacceptblob'].data()
            if not result.has_key(cmslsnum):
                result[cmslsnum]=[]
            result[cmslsnum].extend([prescaleblob,hltcountblob,hltacceptblob])
        del qHandle
        return (runnum,result)
    except Exception,e:
        raise RuntimeError(' dataDML.hltLSById: '+str(e))
def guessDataIdByRun(schema,runnum):
    '''
    get dataids by runnumber, if there are duplicates, pick max(dataid).Bypass full version lookups
    result [lumidataid(0),trgdataid(1),hltdataid(2)] 
    '''
    lumiids=[]
    trgids=[]
    hltids=[]
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.lumidataTablename())
        qHandle.addToTableList(nameDealer.trgdataTablename())
        qHandle.addToTableList(nameDealer.hltdataTablename())
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
        del qHandle
        return [max(lumiids),max(trgids),max(hltids)]
    except Exception,e :
        raise RuntimeError(' dataDML.guessDataIdByRun: '+str(e))
    
def guessnormDataIdByName(schema,normname):
    '''
    get norm dataids by name, if there are duplicates, pick max(dataid).Bypass full version lookups
    select luminorm.data_id from luminorm
    result [luminormdataid{0),luminormdataid(1)]
    '''   
    luminormids=[]
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList( nameDealer.entryTableName(nameDealer.luminormTablename()) )
        qHandle.addToTableList( nameDealer.luminormTablename() )
        qHandle.addToOutputList('DATA_ID','normdataid')
        qConditionStr='NAME=:normname '
        qCondition=coral.AttributeList()
        qCondition.extend('normname','string')
        qCondition['normname'].setData(normname)
        qResult=coral.AttributeList()
        qResult.extend('normdataid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            luminormids.append(normdataid)
        del qHandle
        return max(normdataid)
    except Exception,e :
        raise RuntimeError(' dataDML.guessnormDataIdByName: '+str(e))
########
########
def dataentryIdByRun(schema,runnum,branchfilter):
    '''
    select el.entry_id,et.entry_id,eh.entry_id,el.revision_id,et.revision_id,eh.revision_id from lumidataentiries el,trgdataentries et,hltdataentries eh where el.name=et.name and et.name=eh.name and el.name=:entryname;
    check on entryrev
   
    return [lumientryid,trgentryid,hltentryid]
    '''
    result=[]
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(nameDealer.entryTableName(lumidataTablename()),'el')
        qHandle.addToTableList(nameDealer.entryTableName(trgdataTablename()),'et')
        qHandle.addToTableList(nameDealer.entryTableName(hltdataTablename()),'eh')
        qHandle.addToOutputList('el.ENTRY_ID','lumientryid')
        qHandle.addToOutputList('et.ENTRY_ID','trgentryid')
        qHandle.addToOutputList('eh.ENTRY_ID','hltentryid')
        qConditionStr='el.NAME=et.NAME and et.NAME=eh.NAME and el.NAME=:runnumstr '
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
        del qHandle
        return result
    except Exception,e:
        raise RuntimeError(' dataDML.dataentryIdByRun: '+str(e))
    

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
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(revmaptablename,'rl')
        qHandle.addToTableList(datatablename,'l')
        qHandle.addToOutputList('l.DATA_ID','dataid')
        qHandle.addToOutputList('rl.REVISION_ID','revisionid')
        qConditionStr='l.DATA_ID=rl.DATA_ID and l.ENTRY_ID=:entryid'
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
        return max(dataids)
    except Exception,e:
        raise RuntimeError(' dataDML.latestdataIdByEntry: '+str(e) )

def normdataIdByName(schema,normname,branchfilter):
    '''
    select en.entry_id,en.revision_id from luminormsentries en where en.entry_id=n.entry_id and en.name=:normname
    check revision_id in branchfilter,get entry_id
    select rn.revision_id,n.data_id from luminormrevision rn,luminorms where rn.data_id=n.data_id and n.entry_id=:normentryid
    return max(data_id)
    '''    
    pass

#=======================================================
#   INSERT requires in update transaction
#=======================================================
def addNormToBranch(schema,normname,defaultnorm,optionalnormdata,branchinfo):
    '''
    input:
       defaultvalue: float
       optionalnormdata {'norm_1':norm_1,'energy_1':energy_1,'norm_2':norm_2,'energy_2':energy_2}
    output:
       [normname,revision_id,entry_id,data_id]
    '''
    norm_1=None
    if optionalnormdata.has_key('norm_1'):
        norm_1=norm_1
    energy_1=None
    if optionalnormdata.has_key('energy_1'):
        energy_1=energy_1
    norm_2=None
    if optionalnormdata.has_key('norm_2'):
        norm_2=norm_2
    energy_2=None
    if optionalnormdata.has_key('energy_2'):
        energy_2=energy_2
    try:
        entry_id=revisionDML.entryInBranch(schema,nameDealer.luminormTableName(),normname,branchinfo[1])
        if entry_id is None:
            (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.luminormTableName())
            entryinfo=(revision_id,entry_id,normname,data_id)
            revisionDML.addEntry(schema,nameDealer.luminormTableName(),entryinfo,branchinfo)
        else:
            (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.luminormTableName() )
            revisionDML.addRevision(schema,nameDealer.luminormTableName(),(revision_id,data_id),branchinfo)
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','DEFAULTNORM':'float','NORM_1':'float','ENERGY_1':'float','NORM_2':'float','ENERGY_2':'float'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':normname,'DEFAULTNORM':defaultnorm,'NORM_1':norm_1,'ENERGY_1':energy_1,'NORM_2':norm_2,'ENERGY_2':energy_2}
        db=dbUtil.dbUtil(schema)
        db.insertOneRow(nameDealer.luminormTableName(),tabrowDefDict,tabrowValueDict)
        return [revision_id,entry_id,data_id]
    except :
        raise     
def addTrgRunDataToBranch(schema,runnumber,trgrundata,branchinfo):
    '''
    input:
       trgrundata [bitnames(0),datasource(1)]
       bitnames clob, bitnames separated by ','
    output:
       [runnumber,revision_id,entry_id,data_id]
    '''
    try:   #fixme: need to consider revision only case
        bulkvalues=[]
        bitnames=trgrundata[0]
        bitzeroname=bitnames.split(',')[0]
        datasource=trgrundata[1]
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
        return [runnumber,revision_id,entry_id,data_id]
    except :
        raise    
def addTrgLSData(schema,trglsdata,runnumber,data_id):
    '''
    insert trg per-LS data for given run and data_id, this operation can be split in transaction chuncks 
    input:
        trglsdata {cmslsnum:[deadtime,bitzeroname,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}
    result nrows inserted
    if nrows==0, then this insertion failed
    '''
    try:
        nrows=0
        bulkvalues=[]   
        lstrgDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('CMSLSNUM','unsigned int'),('DEADTIMECOUNT','unsigned long long'),('BITZEROCOUNT','unsigned int'),('BITZEROPRESCALE','unsigned int'),('PRESCALEBLOB','blob'),('TRGCOUNTBLOB','blob')]
        for cmslsnum,perlstrg in trglsdata.items():
            deadtimecount=perlstrg[0]
            bitzeroname=perlstrg[1]
            bitzerocount=perlstrg[2]
            bitzeroprescale=perlstrg[3]
            trgcountblob=perlstrg[4]
            trgprescaleblob=perlstrg[5]
            bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('CMSLSNUM',cmslsnum),('DEADTIMECOUNT',deadtimecount),('BITZEROCOUNT',bitzerocount),('BITZEROPRESCALE',bitzeroprescale),('PRESCALEBLOB',trgprescaleblob),('TRGCOUNTBLOB',trgcountblob)])
        db.bulkInsert(nameDealer.lstrgTableName(),lstrgDefDict,bulkvalues)
        nrows=len(bulkvalues)
        return nrows
    except Exception,e :
        raise RuntimeError(' dataDML.addTrgLSData: '+str(e))

def addHLTRunDataToBranch(schema,runnumber,hltrundata,branchinfo):
    '''
    input:
        hltrundata [pathnameclob(0),datasource(1)]
    output:
        [runnumber,revision_id,entry_id,data_id]
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
         return [runnumber,revision_id,entry_id,data_id]
    except :
        raise 
    
def addHltLSData(schema,hltlsdata,runnumber,data_id):
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
        db.bulkInsert(nameDealer.lshltTableName(),lshltDefDict,bulkvalues)
        return len(bulkvalues)
    except Exception,e :
        raise RuntimeError(' dataDML.addHltLSData: '+str(e))

def addLumiRunDataToBranch(schema,runnumber,lumirundata,branchinfo):
    '''
    input:
          lumirundata [datasource]
    output:
          [runnumber,revision_id,entry_id,data_id]
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
         return [runnumber,revision_id,entry_id,data_id]
    except :
        raise
    
def addLumiLSSummary(schema,lumilsdata,runnumber,data_id):
    '''
    input:
          lumilsdata {lumilsnum:[cmslsnum,cmsalive,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexindexblob,beam1intensity,beam2intensity]}
    output:
          nrows
    '''
    try:
        nrow=0
        bulkvalues=[]
        lslumiDefDict=[('LUMIDATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('LUMILSNUM','unsigned int'),('CMSLSNUM','unsigned int'),('CMSALIVE','short'),('INSTLUMI','float'),('INSTLUMIERROR','float'),('INSTLUMIQUALITY','short'),('BEAMSTATUS','string'),('BEAMENERGY','float'),('CMSBXINDEXBLOB','blob'),('BEAMINTENSITYBLOB_1','blob'),('BEAMINTENSITYBLOB_2','blob'),('NUMORBIT','unsigned int'),('STARTORBIT','unsigned int')]
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
            bulkvalues.append([('LUMIDATA_ID',data_id),('RUNNUM',runnumber),('LUMILSNUM',lumilsnum),('CMSLSNUM',cmslsnum),('CMSALIVE',cmsalive),('INSTLUMI',instlumi),('INSTLUMIERROR',instlumierror),('INSTLUMIQUALITY',instlumiquality),('BEAMSTATUS',beamstatus),('CMSBXINDEXBLOB',beam1intensity),('BEAMINTENSITYBLOB_1',beam1intensity),('BEAMINTENSITYBLOB_2',beam2intensity),('NUMORBIT',numorbit),('STARTORBIT',startorbit)])
        db.bulkInsert(nameDealer.lumisummaryTableName(),lshlumiDefDict,bulkvalues)
        return len(bulkvalues)
    except Exception,e :
        raise RuntimeError(' dataDML.addHltLSData: '+str(e))

def addLumiLSDetail(schema,lumibxdata,runnumber,data_id):
    '''
    input:
          lumibxdata [[algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]}],[algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]}],[algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]}]]
    output:
          nrows
    '''
    try:
        nrow=0
        bulkvalues=[]
        lslumiDefDict=[('LUMIDATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('LUMILSNUM','unsigned int'),('CMSLSNUM','unsigned int'),('CMSALIVE','short'),('INSTLUMI','float'),('INSTLUMIERROR','float'),('INSTLUMIQUALITY','short'),('BEAMSTATUS','string'),('BEAMENERGY','float'),('CMSBXINDEXBLOB','blob'),('BEAMINTENSITYBLOB_1','blob'),('BEAMINTENSITYBLOB_2','blob'),('NUMORBIT','unsigned int'),('STARTORBIT','unsigned int')]
        for peralgobx in lumibxdata:
            algoname=peralgobx[0]
            peralgobxdata=peralgobx[1]
            for lumilsnum,bxdata in peralgobxdata.items():
                cmslsnum=bxdata[0]
                bxlumivalue=bxdata[1]
                bxlumierror=bxdata[2]
                bxlumiquality=bxdata[3]
                bulkvalues.append([('LUMIDATA_ID',data_id),('RUNNUM',runnumber),('LUMILSNUM',lumilsnum),('CMSLSNUM',cmslsnum),('ALGONAME',algoname),('BXLUMIVALUE',bxlumivalue),('BXLUMIERROR',bxlumierror),('BXLUMIQUALITY',bxlumiquality)])
        db.bulkInsert(nameDealer.lumidetailTableName(),lslumiDefDict,bulkvalues)
        return len(bulkvalues)
    except Exception,e :
        raise RuntimeError(' dataDML.addLumiLSDetail: '+str(e))
    
def completeOldLumiData(schema,runnumber,lsdata,data_id):
    '''
    input:
    lsdata [[lumisummary_id,lumilsnum,cmslsnum]]
    '''
    try:
        #update in lumisummary table
        print 'insert in lumisummary table'
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
    except Exception,e :
        raise RuntimeError(' dataDML.completeOldLumiData: '+str(e))
    
#=======================================================
#   DELETE
#=======================================================

if __name__ == "__main__":
    import sessionManager
    import lumidbDDL,revisionDML
    myconstr='sqlite_file:test.db'
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
    addNormToBranch(schema,'pp7TeV',6370,{},normbranchinfo)
    addNormToBranch(schema,'hi7TeV',2.38,{},normbranchinfo)
    (branchid,branchparent)=revisionDML.branchInfoByName(schema,'DATA')
    branchinfo=(branchid,'DATA')
    for runnum in [1200,1211,1222,1233,1345]:
        lumirundata=['dummyroot'+str(runnum)+'.root']
        addLumiRunDataToBranch(schema,runnum,lumirundata,branchinfo)
        trgrundata=['ZeroBias,Pippo,Pippa,Mu15','oracle://cms_orcon_prod/cms_trg']
        addTrgRunDataToBranch(schema,runnum,trgrundata,branchinfo)
        hltrundata=['NewHLTPrescale1,NewHLTPrescale2,HLTJet15U','oracle://cms_orcon_prod/cms_runinfo']
        addHLTRunDataToBranch(schema,runnum,hltrundata,branchinfo)
    session.transaction().commit()
    print 'test reading'
    session.transaction().start(True)
    print '===inspecting NORM branch==='
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
        lumirundata=['dummyroot'+str(runnum)+'.root']
        addLumiRunDataToBranch(schema,runnum,lumirundata,branchinfo)
        trgrundata=['ZeroBias,Pippo,Pippa,Mu15','oracle://cms_orcon_prod/cms_trg']
        addTrgRunDataToBranch(schema,runnum,trgrundata,branchinfo)
        hltrundata=['NewHLTPrescale1,NewHLTPrescale2,HLTJet15U','oracle://cms_orcon_prod/cms_runinfo']
        addHLTRunDataToBranch(schema,runnum,hltrundata,branchinfo)
    revlist=revisionDML.revisionsInTag(schema,revisionid,branchinfo[0])
    print 'revisions in branch DATA',revisionDML.revisionsInBranch(schema,branchinfo[0])
    session.transaction().commit()
    print 'revisions in tag data_orig ',revlist
    del session

#!/usr/bin/env python
VERSION='1.02'
import os,sys,array
import coral
from RecoLuminosity.LumiDB import argparse,idDealer,nameDealer,CommonUtil,dbUtil

class newSchemaNames(object):
    def __init__(self):
        self.revisionstable='REVISIONS'
        self.luminormstable='LUMINORMS'
        self.lumidatatable='LUMIDATA'
        self.lumisummarytable='LUMISUMMARY'
        self.runsummarytable='CMSRUNSUMMARY'
        self.lumidetailtable='LUMIDETAIL'
        self.trgdatatable='TRGDATA'
        self.lstrgtable='LSTRG'
        self.hltdatatable='HLTDATA'
        self.lshlttable='LSHLT'
        self.trghltmaptable='TRGHLTMAP'
        self.validationtable='LUMIVALIDATION'
    def idtablename(self,datatablename):
        return datatablename.upper()+'_ID'
    def entrytablename(self,datatablename):
        return datatablename.upper()+'_ENTRIES'
    def revtablename(self,datatablename):
        return datatablename.upper()+'_REV'
    
class oldSchemaNames(object):
    def __init__(self):
        self.lumisummarytable='LUMISUMMARY'
        self.lumidetailtable='LUMIDETAIL'
        self.runsummarytable='CMSRUNSUMMARY'
        self.trgtable='TRG'
        self.hlttable='HLT'
        self.trghltmaptable='TRGHLTMAP'

def intlistToRange(mylist):
    '''
    [1,2,3,4,5] ->[(1,5)]
    [1] ->[(1,1)]
    [1,2,3,5,6,7,8,9,13,15] - >[(1,3),(5,9),(13,13),(15,15)]
    '''
    result=[]
    low=mylist[0]
    high=x.low
    for i,j in CommonUtil.pairwise(mylist):
        if i+1==j:
            x.high=j
        else:
            if j is None:
                result.append((low,i))
                break
            high=i
            result.append((low,high))
            low=j
    return result

def getBranchRangeById(schema,mybranch_id):
    '''
    select revision_id from revisions where branch_id=:branch_id 
    '''
    result=[]
    mylist=[]
    dbsession.transaction().start(True)
    qHandle=dbsession.nominalSchema().newQuery()
    qHandle.addToTableList(nameDealer.revisiontableName())
    qHandle.addToOutputList('REVISION_ID','revision_id')
    qCondition=coral.AttributeList()
    qCondition.extend('branch_id','unsigned long long')
    qCondition['branch_id'].setData(mybranch_id)
    qResult=coral.AttributeList()
    qResult.extend('revision_id','unsigned long long')
    qHandle.defineOutput(qResult)
    qHandle.setCondition('BRANCH_ID=:branch_id',qCondition)
    cursor=qHandle.execute()
    while cursor.next():
        revision_id=cursor.currentRow()['revision_id'].data()
        mylist.append(revision_id)
    del qHandle
    result=intlistToRange(mylist)
    return result

def isOldSchema(dbsession):
    '''
    if there is no lumidata table, then it is old schema
    '''
    n=newSchemaNames()
    result=False
    dbsession.transaction().start(True)
    db=dbUtil.dbUtil(dbsession.nominalSchema())
    result=db.tableExists(n.lumidatatable)
    dbsession.transaction().commit()
    return not result

def createNewTables(dbsession):
    '''
    create new tables if not exist
    revisions,revisions_id,luminorms,luminorms_entries,luminorms_entries_id,
    '''
    n=newSchemaNames()
    try:        
        dbsession.transaction().start(False)
        dbsession.typeConverter().setSqlTypeForCppType('NUMBER(10)','unsigned int')
        dbsession.typeConverter().setSqlTypeForCppType('NUMBER(20)','unsigned long long')
        schema=dbsession.nominalSchema()
        db=dbUtil.dbUtil(schema)
        
        print 'creating revisions table'
        revisionsTab=coral.TableDescription()
        revisionsTab.setName( n.revisionstable )
        revisionsTab.insertColumn( 'REVISION_ID','unsigned long long')
        revisionsTab.insertColumn( 'BRANCH_ID','unsigned long long')
        revisionsTab.insertColumn( 'NAME', 'string')
        revisionsTab.insertColumn( 'COMMENT', 'string')
        revisionsTab.insertColumn( 'CTIME', 'time stamp',6)
        revisionsTab.setPrimaryKey( 'REVISION_ID' )
        db.createTable(revisionsTab,withIdTable=True)
        
        print 'creating luminorms table'
        luminormsTab=coral.TableDescription()
        luminormsTab.setName( n.luminormstable )
        luminormsTab.insertColumn( 'DATA_ID','unsigned long long')
        luminormsTab.insertColumn( 'ENTRY_ID','unsigned long long')
        luminormsTab.insertColumn( 'DEFAULTNORM', 'float')
        luminormsTab.insertColumn( 'NORM_1', 'float')
        luminormsTab.insertColumn( 'ENERGY_1', 'float')
        luminormsTab.insertColumn( 'NORM_2', 'float')
        luminormsTab.insertColumn( 'ENERGY_2', 'float')
        luminormsTab.setPrimaryKey( 'DATA_ID' )
        db.createTable(luminormsTab,withIdTable=True,withEntryTables=True,withRevMapTable=True)

        print 'creating lumidata table'
        lumidataTab=coral.TableDescription()
        lumidataTab.setName( n.lumidatatable )
        lumidataTab.insertColumn( 'DATA_ID','unsigned long long')
        lumidataTab.insertColumn( 'ENTRY_ID','unsigned long long')
        lumidataTab.insertColumn( 'SOURCE', 'string')
        lumidataTab.insertColumn( 'RUNNUM', 'unsigned int')
        lumidataTab.setPrimaryKey( 'DATA_ID' )
        db.createTable(lumidataTab,withIdTable=True,withEntryTables=True,withRevMapTable=True)

        print 'creating trgdata table'
        trgdataTab=coral.TableDescription()
        trgdataTab.setName( n.trgdatatable )
        trgdataTab.insertColumn( 'DATA_ID','unsigned long long')
        trgdataTab.insertColumn( 'ENTRY_ID','unsigned long long')
        trgdataTab.insertColumn( 'SOURCE', 'string')
        trgdataTab.insertColumn( 'RUNNUM', 'unsigned int')
        trgdataTab.insertColumn( 'BITZERONAME', 'string')
        trgdataTab.insertColumn( 'BITNAMECLOB', 'string',6000)
        trgdataTab.setPrimaryKey( 'DATA_ID' )
        db.createTable(trgdataTab,withIdTable=True,withEntryTables=True,withRevMapTable=True)

        print 'creating lstrg table'
        lstrgTab=coral.TableDescription()
        lstrgTab.setName( n.lstrgtable )
        lstrgTab.insertColumn( 'DATA_ID','unsigned long long')
        lstrgTab.insertColumn( 'RUNNUM', 'unsigned int')
        lstrgTab.insertColumn( 'CMSLSNUM', 'unsigned int')
        lstrgTab.insertColumn( 'DEADTIMECOUNT', 'unsigned long long')
        lstrgTab.insertColumn( 'BITZEROCOUNT', 'unsigned int')
        lstrgTab.insertColumn( 'BITZEROPRESCALE', 'unsigned int')
        lstrgTab.insertColumn( 'DEADFRAC', 'float')
        lstrgTab.insertColumn( 'PRESCALEBLOB', 'blob')
        lstrgTab.insertColumn( 'TRGCOUNTBLOB', 'blob')
        db.createTable(lstrgTab,withIdTable=False)

        print 'creating hltdata table'
        hltdataTab=coral.TableDescription()
        hltdataTab.setName( n.hltdatatable )
        hltdataTab.insertColumn( 'DATA_ID','unsigned long long')
        hltdataTab.insertColumn( 'ENTRY_ID','unsigned long long')
        hltdataTab.insertColumn( 'RUNNUM', 'unsigned int')
        hltdataTab.insertColumn( 'SOURCE', 'string')
        hltdataTab.insertColumn( 'NPATH', 'unsigned int')
        hltdataTab.insertColumn( 'PATHNAMECLOB', 'string',6000)
        hltdataTab.setPrimaryKey( 'DATA_ID' )
        db.createTable(hltdataTab,withIdTable=True,withEntryTables=True,withRevMapTable=True)

        print 'create lshlt table'
        lshltTab=coral.TableDescription()
        lshltTab.setName( n.lshlttable )
        lshltTab.insertColumn( 'DATA_ID','unsigned long long')
        lshltTab.insertColumn( 'RUNNUM', 'unsigned int')
        lshltTab.insertColumn( 'CMSLSNUM', 'unsigned int')
        lshltTab.insertColumn( 'PRESCALEBLOB', 'blob')
        lshltTab.insertColumn( 'HLTCOUNTBLOB', 'blob')
        lshltTab.insertColumn( 'HLTACCEPTBLOB', 'blob')
        db.createTable(lshltTab,withIdTable=False)
        
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.createNewTables: '+str(e))
    

def modifyOldTables(dbsession):
    '''
    modify old tables:lumisummary,lumidetail
    alter table lumisummary add column(data_id unsigned long long)
    alter table lumidetail add column(data_id unsigned long long,runnum unsigned int,cmslsnum unsigned int)
    '''
    n=newSchemaNames()
    try:
        dbsession.transaction().start(False)
        dbsession.typeConverter().setCppTypeForSqlType('unsigned int','NUMBER(10)')
        dbsession.typeConverter().setCppTypeForSqlType('unsigned long long','NUMBER(20)')
        tableHandle=dbsession.nominalSchema().tableHandle(n.lumisummarytable)
        tableHandle.schemaEditor().insertColumn('DATA_ID','unsigned long long')
        tableHandle=dbsession.nominalSchema().tableHandle(n.lumidetailtable)
        tableHandle.schemaEditor().insertColumn('DATA_ID','unsigned long long')
        tableHandle.schemaEditor().insertColumn('RUNNUM','unsigned int')
        tableHandle.schemaEditor().insertColumn('CMSLSNUM','unsigned int')
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.modifyOldTables: '+str(e))
    
def dropNewTables(dbsession):
    n=newSchemaNames()
    try:
        dbsession.transaction().start(False)
        schema=dbsession.nominalSchema()
        db=dbUtil.dbUtil(schema)
        db.dropTable( n.revisionstable )
        
        db.dropTable( n.lumidatatable )
        db.dropTable( n.lumidatatable+'_ID' )
        db.dropTable( n.lumidatatable+'_ENTRIES' )
        db.dropTable( n.lumidatatable+'_ENTRIES_ID' )
        db.dropTable( n.lumidatatable+'_REV' )
        
        db.dropTable( n.luminormstable )
        db.dropTable( n.luminormstable+'_ID' )
        db.dropTable( n.luminormstable+'_ENTRIES' )
        db.dropTable( n.luminormstable+'_ENTRIES_ID' )
        db.dropTable( n.luminormstable+'_REV' )
        
        db.dropTable( n.trgdatatable )
        db.dropTable( n.trgdatatable+'_ID' )
        db.dropTable( n.trgdatatable+'_ENTRIES' )
        db.dropTable( n.trgdatatable+'_ENTRIES_ID' )
        db.dropTable( n.trgdatatable+'_REV' )
        
        db.dropTable( n.hltdatatable )
        db.dropTable( n.hltdatatable+'_ID' )
        db.dropTable( n.hltdatatable+'_ENTRIES' )
        db.dropTable( n.hltdatatable+'_ENTRIES_ID' )
        db.dropTable( n.hltdatatable+'_REV' )
        
        db.dropTable( n.lstrgtable )
        db.dropTable( n.lshlttable )
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.dropNewTables: '+str(e))
    
def restoreOldTables(dbsession):
    n=newSchemaNames()
    try:
        dbsession.transaction().start(False)
        schema=dbsession.nominalSchema()
        dbsession.transaction().start(False)
        tableHandle=dbsession.nominalSchema().tableHandle(n.lumisummarytable)
        tableHandle.schemaEditor().dropColumn('DATA_ID')
        tableHandle=dbsession.nominalSchema().tableHandle(n.lumidetailtable)
        tableHandle.schemaEditor().dropColumn('DATA_ID')
        tableHandle.schemaEditor().dropColumn('RUNNUM')
        tableHandle.schemaEditor().dropColumn('CMSLSNUM')
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.restoreOldTables: '+str(e))
    
def createNewSchema(dbsession):
    '''
    create extra new tables+old unchanged tables
    '''
    createNewTables(dbsession)
    modifyOldTables(dbsession)

def dropNewSchema(dbsession):
    '''
    drop extra new tables+undo column changes
    '''
    dropNewTables(dbsession)
    restoreOldTables(dbsession)

def getOldTrgData(dbsession,runnum):
    '''
    generate new data_id for trgdata
    select cmslsnum,deadtime,bitname,trgcount,prescale from trg where runnum=:runnum and bitnum=0 order by cmslsnum;
    select cmslsnum,bitnum,trgcount,deadtime,prescale from trg where runnum=:runnum order by cmslsnum
    output [bitnames,databuffer]
    '''
    bitnames=''
    databuffer={} #{cmslsnum:[deadtime,bitzeroname,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}
    dbsession.typeConverter().setCppTypeForSqlType('unsigned int','NUMBER(10)')
    dbsession.typeConverter().setCppTypeForSqlType('unsigned long long','NUMBER(20)')
    try:
        dbsession.transaction().start(True)
        qHandle=dbsession.nominalSchema().newQuery()
        n=oldSchemaNames()
        qHandle.addToTableList(n.trgtable)
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('DEADTIME','deadtime')
        qHandle.addToOutputList('BITNAME','bitname')
        qHandle.addToOutputList('TRGCOUNT','trgcount')
        qHandle.addToOutputList('PRESCALE','prescale')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition.extend('bitnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qCondition['bitnum'].setData(int(0))
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('deadtime','unsigned long long')
        qResult.extend('bitname','string')
        qResult.extend('trgcount','unsigned int')
        qResult.extend('prescale','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum AND BITNUM=:bitnum',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            deadtime=cursor.currentRow()['deadtime'].data()
            bitname=cursor.currentRow()['bitname'].data()
            bitcount=cursor.currentRow()['trgcount'].data()
            prescale=cursor.currentRow()['prescale'].data()
            if not databuffer.has_key(cmslsnum):
                databuffer[cmslsnum]=[]
            databuffer[cmslsnum].append(deadtime)
            databuffer[cmslsnum].append(bitname)
            databuffer[cmslsnum].append(bitcount)
            databuffer[cmslsnum].append(prescale)
        del qHandle
        qHandle=dbsession.nominalSchema().newQuery()
        qHandle.addToTableList(n.trgtable)
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('BITNUM','bitnum')
        qHandle.addToOutputList('BITNAME','bitname')
        qHandle.addToOutputList('TRGCOUNT','trgcount')
        qHandle.addToOutputList('PRESCALE','prescale')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qHandle.addToOrderList('cmslsnum')
        qHandle.addToOrderList('bitnum')
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('bitnum','unsigned int')
        qResult.extend('bitname','string')
        qResult.extend('trgcount','unsigned int')
        qResult.extend('prescale','unsigned int')
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        bitnameList=[]
        trgcountArray=array.array('l')
        prescaleArray=array.array('l')
        counter=0
        previouscmslsnum=0
        cmslsnum=-1
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            bitnum=cursor.currentRow()['bitnum'].data()
            bitname=cursor.currentRow()['bitname'].data()
            trgcount=cursor.currentRow()['trgcount'].data()        
            prescale=cursor.currentRow()['prescale'].data()
            
            if bitnum==0 and counter!=0:
                trgcountBlob=CommonUtil.packArraytoBlob(trgcountArray)
                prescaleBlob=CommonUtil.packArraytoBlob(prescaleArray)
                databuffer[previouslsnum].append(trgcountBlob)
                databuffer[previouslsnum].append(prescaleBlob)
                bitnameList=[]
                trgcountArray=array.array('l')
                prescaleArray=array.array('l')
            else:
                previouslsnum=cmslsnum
            bitnameList.append(bitname)
            trgcountArray.append(trgcount)
            prescaleArray.append(prescale)
            counter+=1
        if cmslsnum>0:
            bitnames=','.join(bitnameList)
            trgcountBlob=CommonUtil.packArraytoBlob(trgcountArray)
            prescaleBlob=CommonUtil.packArraytoBlob(prescaleArray)

            databuffer[cmslsnum].append(trgcountBlob)
            databuffer[cmslsnum].append(prescaleBlob)
        del qHandle
        dbsession.transaction().commit()
        return [bitnames,databuffer]
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.getOldTrgData: '+str(e))
    
def transfertrgData(dbsession,runnumber,trgrawdata,branchName='TRGDATA'):
    '''
    input: trgdata [bitnames,databuffer], databuffer {cmslsnum:[deadtime,bitzeroname,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}
    '''
    bulkvalues=[]
    bitzeroname=trgrawdata[0].split(',')[0]
    perlsrawdatadict=trgrawdata[1]
    try:
        dbsession.transaction().start(False)
        (revision_id,entry_id,data_id)=bookNewEntry(dbsession.nominalSchema(),nameDealer.trgdataTableName())
        #print 'revision_id,entry_id,data_id ',revision_id,entry_id,data_id
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','RUNNUM':'unsigned int','BITZERONAME':'string','BITNAMECLOB':'string'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'RUNNUM':int(runnumber),'BITZERONAME':bitzeroname,'BITNAMECLOB':trgrawdata[0]}
        db=dbUtil.dbUtil(dbsession.nominalSchema())
        db.insertOneRow(nameDealer.trgdataTableName(),tabrowDefDict,tabrowValueDict)
#        addEntry(dbsession.nominalSchema(),nameDealer.trgdataTableName(),revision_id,entry_id,data_id)        
        lstrgDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('CMSLSNUM','unsigned int'),('DEADTIMECOUNT','unsigned long long'),('BITZEROCOUNT','unsigned int'),('BITZEROPRESCALE','unsigned int'),('PRESCALEBLOB','blob'),('TRGCOUNTBLOB','blob')]
        for cmslsnum,perlstrg in perlsrawdatadict.items():
            deadtimecount=perlstrg[0]
            bitzeroname=perlstrg[1]
            bitzerocount=perlstrg[2]
            bitzeroprescale=perlstrg[3]
            trgcountblob=perlstrg[4]
            trgprescaleblob=perlstrg[5]
            bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('CMSLSNUM',cmslsnum),('DEADTIMECOUNT',deadtimecount),('BITZEROCOUNT',bitzerocount),('BITZEROPRESCALE',bitzeroprescale),('PRESCALEBLOB',trgprescaleblob),('TRGCOUNTBLOB',trgcountblob)])
        db.bulkInsert(nameDealer.lstrgTableName(),lstrgDefDict,bulkvalues)
        addEntryToBranch(dbsession.nominalSchema(),nameDealer.trgdataTableName(),revision_id,entry_id,data_id,branchName,str(runnumber),'transfer2010')
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.transfertrgData: '+str(e))

def transferhltData(dbsession,runnumber,hltrawdata,branchName='HLTDATA'):
    '''
    input: hltdata[pathnames,databuffer] #databuffer{cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}
    '''
    npath=len(hltrawdata[0].split(','))
    pathnames=hltrawdata[0]
    perlsrawdatadict=hltrawdata[1]
    bulkvalues=[]
    try:
        dbsession.transaction().start(False)
        (revision_id,entry_id,data_id)=bookNewEntry(dbsession.nominalSchema(),nameDealer.hltdataTableName())
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','RUNNUM':'unsigned int','NPATH':'unsigned int','PATHNAMECLOB':'string'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'RUNNUM':int(runnumber),'NPATH':npath,'PATHNAMECLOB':pathnames}
        db=dbUtil.dbUtil(dbsession.nominalSchema())
        db.insertOneRow(nameDealer.hltdataTableName(),tabrowDefDict,tabrowValueDict)
        lshltDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('CMSLSNUM','unsigned int'),('PRESCALEBLOB','blob'),('HLTCOUNTBLOB','blob'),('HLTACCEPTBLOB','blob')]
        for cmslsnum,perlshlt in perlsrawdatadict.items():
            inputcountblob=perlshlt[0]
            acceptcountblob=perlshlt[1]
            prescaleblob=perlshlt[2]
            bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('CMSLSNUM',cmslsnum),('PRESCALEBLOB',prescaleblob),('HLTCOUNTBLOB',inputcountblob),('HLTACCEPTBLOB',acceptcountblob)])
        db.bulkInsert(nameDealer.lshltTableName(),lshltDefDict,bulkvalues)
        addEntryToBranch(dbsession.nominalSchema(),nameDealer.hltdataTableName(),revision_id,entry_id,data_id,branchName,str(runnumber),'transfer2010')
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.transferhltData: '+str(e))

def bookNewEntry(schema,datatableName):
    '''
    allocate new revision_id,entry_id,data_id
    '''
    n=newSchemaNames()
    entrytableName=nameDealer.entryTableName(datatableName)
    iddealer=idDealer.idDealer(schema)
    revision_id=iddealer.generateNextIDForTable(n.revisionstable)
    data_id=iddealer.generateNextIDForTable(datatableName)
    entry_id=iddealer.generateNextIDForTable(entrytableName)
    return (revision_id,entry_id,data_id)

def bookNewRevision(schema,datatableName):
    '''
    allocate new revision_id,data_id
    '''
    n=newSchemaNames()
    iddealer=idDealer.idDealer(schema)
    revision_id=iddealer.generateNextIDForTable(n.revisionstable)
    data_id=iddealer.generateNextIDForTable(datatableName)
    return (revision_id,data_id)

def addEntryToBranch(schema,datatableName,revision_id,entry_id,data_id,branchname,name='',comment=''):
    (parentrevision_id,parentbranch_id)=getBranchByName(schema,branchname)
    addEntry(schema,datatableName,revision_id,entry_id,data_id,parentrevision_id,name,comment)
     
def addEntry(schema,datatableName,revision_id,entry_id,data_id,branch_id=0,name='',comment=''):
    '''
    1.allocate and insert a new revision into the revisions table
    2.allocate and insert a new entry into the entry table with the new revision
    3.inset into data_rev table with new data_id ,revision)id mapping
    
    insert into revisions(revision_id,branch_id,name,comment,ctime) values()
    insert into datatablename_entries (entry_id,revision_id) values()
    insert into datatablename_rev(data_id,revision_id) values()
    '''
    revisiontableName=nameDealer.revisionTableName()
    entrytableName=nameDealer.entryTableName(datatableName)
    revtableName=nameDealer.revmapTableName(datatableName)
    
    db=dbUtil.dbUtil(schema)
    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['BRANCH_ID']='unsigned long long'
    tabrowDefDict['NAME']='string'
    tabrowDefDict['COMMENT']='string'
    tabrowDefDict['CTIME']='time stamp'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=revision_id
    tabrowValueDict['BRANCH_ID']=branch_id
    tabrowValueDict['NAME']=name
    tabrowValueDict['COMMENT']=comment
    tabrowValueDict['CTIME']=coral.TimeStamp()
    db.insertOneRow(revisiontableName,tabrowDefDict,tabrowValueDict)
    
    tabrowDefDict={}
    tabrowDefDict['ENTRY_ID']='unsigned long long'
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowValueDict={}
    tabrowValueDict['ENTRY_ID']=entry_id
    tabrowValueDict['REVISION_ID']=revision_id
    db.insertOneRow(entrytableName,tabrowDefDict,tabrowValueDict)

    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['DATA_ID']='unsigned long long'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=revision_id
    tabrowValueDict['DATA_ID']=data_id
    db.insertOneRow(revtableName,tabrowDefDict,tabrowValueDict)
    
def addRevision(schema,datatableName,revision_id,data_id,branch_id=0,name='',comment=''):
    '''
    1.insert a new revision into the revisions table
    2.insert into data_id, revision_id pair to  datatable_rev 
    insert into revisions(revision_id,branch_id,name,comment,ctime) values()
    insert into datatable_rev(data_id,revision_id) values())
    '''
    revisiontableName=nameDealer.revisionTableName()
    revtableName=nameDealer.revmapTableName(datatableName)
    
    db=dbUtil.dbUtil(schema)
    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['BRANCH_ID']='unsigned long long'
    tabrowDefDict['NAME']='string'
    tabrowDefDict['COMMENT']='string'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=revision_id
    tabrowValueDict['BRANCH_ID']=branch_id
    tabrowValueDict['NAME']=name
    tabrowValueDict['COMMENT']=comment
    db.insertOneRow(revisiontableName,tabrowDefDict,tabrowValueDict)

    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['DATA_ID']='unsigned long long'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=revision_id
    tabrowValueDict['DATA_ID']=data_id
    db.insertOneRow(revtableName,tabrowDefDict,tabrowValueDict)

def createNewBranch(schema,name,comment='',parentname=None):
    '''
    create a new branch under given parentbranch
    if parentname=None, create branch under root,branch_id=0
    select revision_id from revisions where name=:parentname
    insert into revisions(revision_id,branch_id,name) values()
    '''
    parentrevision_id=None
    if parentname is None:
        parentrevision_id=0
        revision_id=0
    else:
        try:
            qHandle=schema.newQuery()
            qHandle.addToTableList( nameDealer.revisionTableName() )
            qHandle.addToOutputList('REVISION_ID','revision_id')
            qCondition=coral.AttributeList()
            qCondition.extend('parentname','string')
            qCondition['parentname'].setData(parentname)
            qResult=coral.AttributeList()
            qResult.extend('revision_id','unsigned long long')
            qHandle.defineOutput(qResult)
            qHandle.setCondition('NAME=:parentname',qCondition)
            cursor=qHandle.execute()
            while cursor.next():
                parentrevision_id=cursor.currentRow()['revision_id'].data()
            del qHandle
            iddealer=idDealer.idDealer(schema)
            revision_id=iddealer.generateNextIDForTable( nameDealer.revisionTableName() )
        except Exception,er:
            raise RuntimeError(' migrateSchema.createNewBranch: '+str(er))
    if parentrevision_id is None:
        raise RuntimeError(' migrateSchema.createNewBranch: non-existing parent node '+parentname)
        
    db=dbUtil.dbUtil(schema)
    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['BRANCH_ID']='unsigned long long'
    tabrowDefDict['NAME']='string'
    tabrowDefDict['COMMENT']='string'
    tabrowDefDict['CTIME']='time stamp'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=revision_id
    tabrowValueDict['BRANCH_ID']=parentrevision_id
    tabrowValueDict['NAME']=name
    tabrowValueDict['COMMENT']=comment
    tabrowValueDict['CTIME']=coral.TimeStamp()
    db.insertOneRow(nameDealer.revisionTableName(),tabrowDefDict, tabrowValueDict )
    return revision_id

def getBranchByName(schema,branchName):
    '''
    select branch_id from revisions where name=:branchName
    '''
    try:
         qHandle=schema.newQuery()
         qHandle.addToTableList( nameDealer.revisionTableName() )
         qHandle.addToOutputList('REVISION_ID','revision_id')
         qHandle.addToOutputList('BRANCH_ID','branch_id')
         qCondition=coral.AttributeList()
         qCondition.extend('name','string')
         qCondition['name'].setData(branchName)
         qResult=coral.AttributeList()
         qResult.extend('revision_id','unsigned long long')
         qResult.extend('branch_id','unsigned long long')
         qHandle.defineOutput(qResult)
         qHandle.setCondition('NAME=:name',qCondition)
         cursor=qHandle.execute()
         while cursor.next():
             revision_id=cursor.currentRow()['revision_id'].data()
             branch_id=cursor.currentRow()['branch_id'].data()
         del qHandle
         return (revision_id,branch_id)
    except Exception,e :
        raise RuntimeError(' migrateSchema.getBranchByName: '+str(e))
    
def createLumiNorm(dbsession,name,inputdata,branchName='LUMINORM',comment=''):
    '''
    add new lumi norm entry
    inputdata={'defaultnorm':defaultnorm,'norm_1':norm_1,'energy_1':energy_1,'norm_2':norm_2,'energy_2':energy_2}
    '''
    try:
        dbsession.transaction().start(False)
        db=dbUtil.dbUtil(dbsession.nominalSchema())
        (parentrevision_id,parentbranch_id)=getBranchByName(dbsession.nominalSchema(),branchName)
        (revision_id,entry_id,data_id)=bookNewEntry(dbsession.nominalSchema(),nameDealer.luminormTableName())
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','DEFAULTNORM':'float','NORM_1':'float','ENERGY_1':'float','NORM_2':'float','ENERGY_2':'float'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'DEFAULTNORM':inputdata['DEFAULTNORM'],'NORM_1':inputdata['NORM_1'],'ENERGY_1':inputdata['ENERGY_1'],'NORM_2':inputdata['NORM_2'],'ENERGY_2':inputdata['ENERGY_2']}
        db.insertOneRow(nameDealer.luminormTableName(),tabrowDefDict,tabrowValueDict)
        addEntryToBranch(dbsession.nominalSchema(),nameDealer.luminormTableName(),revision_id,entry_id,data_id,branchName,name,comment)
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.createLumiNorm: '+str(e))
    return data_id

def transferLumiData(dbsession,runnum,branchName='LUMIDATA'):
    '''
    select LUMISUMMARY_ID as lumisummary_id,CMSLSNUM as cmslsnum from LUMISUMMARY where RUNNUM=:runnum order by cmslsnum
    generate new data_id for lumidata
    insert into data_id , runnum into lumidata
    insert into data_id into lumisummary
    insert into data_id into lumidetail
    '''
    n=newSchemaNames()
    m=oldSchemaNames()
    lumisummarydata=[]
    try:
        dbsession.transaction().start(True)
        #find lumi_summaryid of given run
        qHandle=dbsession.nominalSchema().newQuery()
        qHandle.addToTableList(n.lumisummarytable)
        qHandle.addToOutputList('LUMISUMMARY_ID','lumisummary_id')
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qResult=coral.AttributeList()
        qResult.extend('lumisummary_id','unsigned long long')
        qResult.extend('cmslsnum','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qHandle.addToOrderList('cmslsnum')
        cursor=qHandle.execute()
        while cursor.next():
            lumisummary_id=cursor.currentRow()['lumisummary_id'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            lumisummarydata.append((lumisummary_id,cmslsnum))
        del qHandle
        dbsession.transaction().commit()
        
        dbsession.transaction().start(False)
        (revision_id,entry_id,data_id)=bookNewEntry(dbsession.nominalSchema(),nameDealer.lumidataTableName())
        print 'insert in lumidata table'
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','RUNNUM':'unsigned int'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'RUNNUM':int(runnum)}
        db=dbUtil.dbUtil(dbsession.nominalSchema())
        db.insertOneRow(nameDealer.lumidataTableName(),tabrowDefDict,tabrowValueDict)
        addEntryToBranch(dbsession.nominalSchema(),nameDealer.lumidataTableName(),revision_id,entry_id,data_id,branchName,str(runnum),'transfer2010')
        #update in lumisummary table
        print 'insert in lumisummary table'
        setClause='DATA_ID=:data_id'
        updateCondition='RUNNUM=:runnum'
        updateData=coral.AttributeList()
        updateData.extend('data_id','unsigned long long')
        updateData.extend('runnum','unsigned int')
        updateData['data_id'].setData(data_id)
        updateData['runnum'].setData(int(runnum))
        print 'about to singleUpdate'
        nrows=db.singleUpdate(n.lumisummarytable,setClause,updateCondition,updateData)
        #updates in lumidetail table
        print 'update to data_id,lumisummary_id,cmslsnum ',data_id,lumisummary_id,cmslsnum
        updateAction='DATA_ID=:data_id,RUNNUM=:runnum,CMSLSNUM=:cmslsnum'
        updateCondition='LUMISUMMARY_ID=:lumisummary_id'
        bindvarDef=[]
        bindvarDef.append(('data_id','unsigned long long'))
        bindvarDef.append(('runnum','unsigned int'))
        bindvarDef.append(('cmslsnum','unsigned int'))
        bindvarDef.append(('lumisummary_id','unsigned long long'))
        inputData=[]
        for (lumisummary_id,cmslsnum) in lumisummarydata:
            inputData.append([('data_id',data_id),('runnum',int(runnum)),('cmslsnum',cmslsnum),('lumisummary_id',lumisummary_id)])
        db.updateRows(n.lumidetailtable,updateAction,updateCondition,bindvarDef,inputData)
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise RuntimeError(' migrateSchema.transferLumiData: '+str(e))
    return data_id

def getOldHLTData(dbsession,runnum):
    '''
    select count(distinct pathname) from hlt where runnum=:runnum
    select cmslsnum,pathname,inputcount,acceptcount,prescale from hlt where runnum=:runnum order by cmslsnum,pathname
    [pathnames,databuffer]
    '''
    
    databuffer={} #{cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}
    dbsession.typeConverter().setCppTypeForSqlType('unsigned int','NUMBER(10)')
    dbsession.typeConverter().setCppTypeForSqlType('unsigned long long','NUMBER(20)')
    pathnames=''
    try:
        npath=0
        dbsession.transaction().start(True)
        qHandle=dbsession.nominalSchema().newQuery()
        n=oldSchemaNames()
        qHandle.addToTableList(n.hlttable)
        qHandle.addToOutputList('COUNT(DISTINCT PATHNAME)','npath')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qResult=coral.AttributeList()
        qResult.extend('npath','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            npath=cursor.currentRow()['npath'].data()
        del qHandle
        #print 'npath ',npath
        dbsession.transaction().start(True)
        qHandle=dbsession.nominalSchema().newQuery()
        n=oldSchemaNames()
        qHandle.addToTableList(n.hlttable)
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('PATHNAME','pathname')
        qHandle.addToOutputList('INPUTCOUNT','inputcount')
        qHandle.addToOutputList('ACCEPTCOUNT','acceptcount')
        qHandle.addToOutputList('PRESCALE','prescale')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('pathname','string')
        qResult.extend('inputcount','unsigned int')
        qResult.extend('acceptcount','unsigned int')
        qResult.extend('prescale','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qHandle.addToOrderList('cmslsnum')
        qHandle.addToOrderList('pathname')
        cursor=qHandle.execute()
        pathnameList=[]
        inputcountArray=array.array('l')
        acceptcountArray=array.array('l')
        prescaleArray=array.array('l')
        ipath=0
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            pathname=cursor.currentRow()['pathname'].data()
            ipath+=1
            inputcount=cursor.currentRow()['inputcount'].data()
            acceptcount=cursor.currentRow()['acceptcount'].data()
            prescale=cursor.currentRow()['prescale'].data()
            pathnameList.append(pathname)
            inputcountArray.append(inputcount)
            acceptcountArray.append(acceptcount)
            prescaleArray.append(prescale)
            if ipath==npath:
                pathnames=','.join(pathnameList)
                inputcountBlob=CommonUtil.packArraytoBlob(inputcountArray)
                acceptcountBlob=CommonUtil.packArraytoBlob(acceptcountArray)
                prescaleBlob=CommonUtil.packArraytoBlob(prescaleArray)
                databuffer[cmslsnum]=[inputcountBlob,acceptcountBlob,prescaleBlob]
                pathnameList=[]
                inputcountArray=array.array('l')
                acceptcountArray=array.array('l')
                prescaleArray=array.array('l')
                ipath=0
        del qHandle
        dbsession.transaction().commit()
    except Exception,e :
        dbsession.transaction().rollback()
        del dbsession
        raise Exception,' migrateSchema.getOldTrgData: '+str(e)
    return [pathnames,databuffer]

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="migrate lumidb schema",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',dest='connect',action='store',required=False,default='oracle://cms_orcoff_prep/CMS_LUMI_DEV_OFFLINE',help='connect string to trigger DB(required)')
    parser.add_argument('-P',dest='authpath',action='store',required=False,default='/afs/cern.ch/user/x/xiezhen',help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',required=True,help='run number')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=int(args.runnumber)
    print 'processing run ',runnumber
    os.environ['CORAL_AUTH_PATH']=args.authpath
    svc=coral.ConnectionService()
    dbsession=svc.connect(args.connect,accessMode=coral.access_Update)
    if args.debug:
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    if isOldSchema(dbsession):
        print 'is old schema'
        createNewSchema(dbsession)
    else:
        print 'is new schema'
        dropNewSchema(dbsession)
        print 'creating new schema'
        createNewSchema(dbsession)
        print 'done'
    dbsession.transaction().start(False)
    createNewBranch(dbsession.nominalSchema(),'TRUNK',comment='root',parentname=None)
    createNewBranch(dbsession.nominalSchema(),'LUMIDATA',comment='root of lumidata',parentname='TRUNK')
    createNewBranch(dbsession.nominalSchema(),'LUMINORM',comment='root of luminorm',parentname='TRUNK')
    createNewBranch(dbsession.nominalSchema(),'TRGDATA',comment='root of trgdata',parentname='TRUNK')
    createNewBranch(dbsession.nominalSchema(),'HLTDATA',comment='root of hltdata',parentname='TRUNK')
    dbsession.transaction().commit()
    normdef={'DEFAULTNORM':6.37,'NORM_1':6.37,'ENERGY_1':3.5e03,'NORM_2':1.625,'ENERGY_2':0.9e03}
    createLumiNorm(dbsession,'pp7TeV',normdef,branchName='LUMINORM')
    trgresult=getOldTrgData(dbsession,runnumber)
    hltresult=getOldHLTData(dbsession,runnumber)
    transferLumiData(dbsession,runnumber,branchName='LUMIDATA')
    transfertrgData(dbsession,runnumber,trgresult,branchName='TRGDATA')
    transferhltData(dbsession,runnumber,hltresult,branchName='HLTDATA')
    del dbsession
    del svc
    #print trgresult[0]
    #print len(trgresult[0].split(','))
    #print trgresult[1]
    #print '==========='
    #print hltresult[0]
    #print len(hltresult[0].split(','))
    #print hltresult[1]
    #print result
if __name__=='__main__':
    main()
    

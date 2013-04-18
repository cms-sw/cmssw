import coral
from RecoLuminosity.LumiDB import nameDealer,dbUtil
#=======================================================
#
#   CREATE
#
#=======================================================
def createTables(schema):
    '''
    create new tables if not exist
    revisions,revisions_id,luminorms,luminorms_entries,luminorms_entries_id,
    '''
    try:
        created=[]
        db=dbUtil.dbUtil(schema)
        if not schema.existsTable(nameDealer.revisionTableName()):
            print 'creating revisions table'
            revisionsTab=coral.TableDescription()
            revisionsTab.setName( nameDealer.revisionTableName() )
            revisionsTab.insertColumn( 'REVISION_ID','unsigned long long')
            revisionsTab.insertColumn( 'BRANCH_ID','unsigned long long')
            revisionsTab.insertColumn( 'NAME', 'string')
            revisionsTab.insertColumn( 'BRANCH_NAME', 'string')
            revisionsTab.insertColumn( 'COMMENT', 'string')
            revisionsTab.insertColumn( 'CTIME', 'time stamp',6)
            revisionsTab.setPrimaryKey( 'REVISION_ID' )
            revisionsTab.setUniqueConstraint(('NAME'))
            db.createTable(revisionsTab,withIdTable=True)
            created.append(nameDealer.revisionTableName())
            
        if not schema.existsTable(nameDealer.luminormTableName()):
            print 'creating luminorms table'
            luminormsTab=coral.TableDescription()
            luminormsTab.setName( nameDealer.luminormTableName() )
            luminormsTab.insertColumn( 'DATA_ID','unsigned long long')
            luminormsTab.insertColumn( 'ENTRY_ID','unsigned long long')
            luminormsTab.insertColumn( 'ENTRY_NAME','string')
            luminormsTab.insertColumn( 'AMODETAG', 'string')
            luminormsTab.insertColumn( 'NORM_1', 'float')
            luminormsTab.insertColumn( 'EGEV_1', 'unsigned int')
            luminormsTab.insertColumn( 'NORM_2', 'float')
            luminormsTab.insertColumn( 'EGEV_2', 'unsigned int')
            luminormsTab.setPrimaryKey( 'DATA_ID' )
            db.createTable(luminormsTab,withIdTable=True,withEntryTables=True,withRevMapTable=True)
            created.append(nameDealer.luminormTableName())
  
        if not schema.existsTable(nameDealer.lumidataTableName()):
            print 'creating lumidata table'
            lumidataTab=coral.TableDescription()
            lumidataTab.setName( nameDealer.lumidataTableName() )
            lumidataTab.insertColumn( 'DATA_ID','unsigned long long')
            lumidataTab.insertColumn( 'ENTRY_ID','unsigned long long')
            lumidataTab.insertColumn( 'ENTRY_NAME','string')
            lumidataTab.insertColumn( 'SOURCE', 'string')
            lumidataTab.insertColumn( 'RUNNUM', 'unsigned int')
            lumidataTab.insertColumn( 'NOMINALEGEV', 'float')
            lumidataTab.setPrimaryKey( 'DATA_ID' )
            db.createTable(lumidataTab,withIdTable=True,withEntryTables=True,withRevMapTable=True)
            created.append(nameDealer.lumidataTableName())

        if not schema.existsTable(nameDealer.lumisummaryv2TableName() ):
            print 'creating lumisummaryv2 table'
            summary=coral.TableDescription()
            summary.setName( nameDealer.lumisummaryv2TableName() )
            summary.insertColumn('DATA_ID','unsigned long long')
            summary.insertColumn('RUNNUM','unsigned int')
            summary.insertColumn('LUMILSNUM','unsigned int')
            summary.insertColumn('CMSLSNUM','unsigned int')
            summary.insertColumn('INSTLUMI','float')
            summary.insertColumn('INSTLUMIERROR','float')
            summary.insertColumn('INSTLUMIQUALITY','short')
            summary.insertColumn('BEAMSTATUS','string')
            summary.insertColumn('BEAMENERGY','float')
            summary.insertColumn('NUMORBIT','unsigned int')
            summary.insertColumn('STARTORBIT','unsigned int')
            summary.insertColumn('CMSBXINDEXBLOB','blob')
            summary.insertColumn('BEAMINTENSITYBLOB_1','blob')
            summary.insertColumn('BEAMINTENSITYBLOB_2','blob')
            summary.insertColumn('BXLUMIVALUE_OCC1','blob')
            summary.insertColumn('BXLUMIVALUE_OCC2','blob')
            summary.insertColumn('BXLUMIVALUE_ET','blob')
            summary.insertColumn('BXLUMIERROR_OCC1','blob')
            summary.insertColumn('BXLUMIERROR_OCC2','blob')
            summary.insertColumn('BXLUMIERROR_ET','blob')
            summary.insertColumn('BXLUMIQUALITY_OCC1','blob')
            summary.insertColumn('BXLUMIQUALITY_OCC2','blob')
            summary.insertColumn('BXLUMIQUALITY_ET','blob')
            summary.setPrimaryKey(('DATA_ID','LUMILSNUM'))
            db.createTable(summary,withIdTable=False)            
            created.append(nameDealer.lumisummaryv2TableName())
            
        #
        # This table exists in the old schema
        #
        if not schema.existsTable(nameDealer.cmsrunsummaryTableName()):
            print 'creating cmsrunsummary table'
            cmsrunsummary=coral.TableDescription()
            cmsrunsummary.setName( nameDealer.cmsrunsummaryTableName() )
            cmsrunsummary.insertColumn('RUNNUM','unsigned int')
            cmsrunsummary.insertColumn('HLTKEY','string')
            cmsrunsummary.insertColumn('L1KEY','string')
            cmsrunsummary.insertColumn('FILLNUM','unsigned int')
            cmsrunsummary.insertColumn('SEQUENCE','string')
            cmsrunsummary.insertColumn('STARTTIME','time stamp',6)
            cmsrunsummary.insertColumn('STOPTIME','time stamp',6)
            cmsrunsummary.insertColumn('EGEV','unsigned int')
            cmsrunsummary.insertColumn('AMODETAG','string')
            cmsrunsummary.setPrimaryKey('RUNNUM')
            db.createTable(cmsrunsummary,withIdTable=False)
            created.append(nameDealer.cmsrunsummaryTableName())
        #
        #  This table exists in the old schema
        #
        if not schema.existsTable(nameDealer.trghltMapTableName()):
            print 'creating trghltmap table'
            trghlt=coral.TableDescription()
            trghlt.setName( nameDealer.trghltMapTableName() )
            trghlt.insertColumn( 'HLTKEY','string' )
            trghlt.insertColumn( 'HLTPATHNAME','string' )
            trghlt.insertColumn( 'L1SEED','string' )
            trghlt.setNotNullConstraint('HLTKEY',True)
            trghlt.setNotNullConstraint('HLTPATHNAME',True)
            trghlt.setNotNullConstraint('L1SEED',True)
            db.createTable(trghlt,withIdTable=False)
            created.append(nameDealer.trghltMapTableName())      
        if not schema.existsTable(nameDealer.trgdataTableName()):
            print 'creating trgdata table'
            trgdataTab=coral.TableDescription()
            trgdataTab.setName( nameDealer.trgdataTableName() )
            trgdataTab.insertColumn( 'DATA_ID','unsigned long long')
            trgdataTab.insertColumn( 'ENTRY_ID','unsigned long long')
            trgdataTab.insertColumn( 'ENTRY_NAME','string')
            trgdataTab.insertColumn( 'SOURCE', 'string')
            trgdataTab.insertColumn( 'RUNNUM', 'unsigned int')
            trgdataTab.insertColumn( 'BITZERONAME', 'string')
            trgdataTab.insertColumn( 'BITNAMECLOB', 'string',6000)
            trgdataTab.setPrimaryKey( 'DATA_ID' )
            db.createTable(trgdataTab,withIdTable=True,withEntryTables=True,withRevMapTable=True)
            created.append(nameDealer.trgdataTableName())
        if not schema.existsTable(nameDealer.lstrgTableName()):
            print 'creating lstrg table'
            lstrgTab=coral.TableDescription()
            lstrgTab.setName( nameDealer.lstrgTableName() )
            lstrgTab.insertColumn( 'DATA_ID','unsigned long long')
            lstrgTab.insertColumn( 'RUNNUM', 'unsigned int')
            lstrgTab.insertColumn( 'CMSLSNUM', 'unsigned int')
            lstrgTab.insertColumn( 'DEADTIMECOUNT', 'unsigned long long')
            lstrgTab.insertColumn( 'BITZEROCOUNT', 'unsigned int')
            lstrgTab.insertColumn( 'BITZEROPRESCALE', 'unsigned int')
            lstrgTab.insertColumn( 'DEADFRAC', 'float')
            lstrgTab.insertColumn( 'PRESCALEBLOB', 'blob')
            lstrgTab.insertColumn( 'TRGCOUNTBLOB', 'blob')
            lstrgTab.setPrimaryKey( ('DATA_ID','CMSLSNUM') )
            db.createTable(lstrgTab,withIdTable=False)            
            created.append( nameDealer.lstrgTableName() )

        if not schema.existsTable(nameDealer.hltdataTableName()):
            print 'creating hltdata table'
            hltdataTab=coral.TableDescription()
            hltdataTab.setName( nameDealer.hltdataTableName() )
            hltdataTab.insertColumn( 'DATA_ID','unsigned long long')
            hltdataTab.insertColumn( 'ENTRY_ID','unsigned long long')
            hltdataTab.insertColumn( 'ENTRY_NAME','string')
            hltdataTab.insertColumn( 'RUNNUM', 'unsigned int')
            hltdataTab.insertColumn( 'SOURCE', 'string')
            hltdataTab.insertColumn( 'NPATH', 'unsigned int')
            hltdataTab.insertColumn( 'PATHNAMECLOB', 'string',6000)
            hltdataTab.setPrimaryKey( 'DATA_ID' )
            db.createTable(hltdataTab,withIdTable=True,withEntryTables=True,withRevMapTable=True)
            created.append(nameDealer.hltTableName())
            
        if not schema.existsTable(nameDealer.lshltTableName()):
            print 'creating lshlt table'
            lshltTab=coral.TableDescription()
            lshltTab.setName( nameDealer.lshltTableName() )
            lshltTab.insertColumn( 'DATA_ID','unsigned long long')
            lshltTab.insertColumn( 'RUNNUM', 'unsigned int')
            lshltTab.insertColumn( 'CMSLSNUM', 'unsigned int')
            lshltTab.insertColumn( 'PRESCALEBLOB', 'blob')
            lshltTab.insertColumn( 'HLTCOUNTBLOB', 'blob')
            lshltTab.insertColumn( 'HLTACCEPTBLOB', 'blob')
            db.createTable(lshltTab,withIdTable=False)
            lshltTab.setPrimaryKey( ('DATA_ID','CMSLSNUM') )
            created.append(nameDealer.lshltTableName())          

            
        if not schema.existsTable(nameDealer.lumivalidationTableName()):
            print 'creating lumivalidation  table'
            lumivalidation=coral.TableDescription()
            lumivalidation.setName( nameDealer.lumivalidationTableName() )
            lumivalidation.insertColumn( 'RUNNUM','unsigned int' )
            lumivalidation.insertColumn( 'CMSLSNUM','unsigned int' )
            lumivalidation.insertColumn( 'FLAG','string' )
            lumivalidation.insertColumn( 'COMMENT','string' )
            lumivalidation.setPrimaryKey( ('RUNNUM','CMSLSNUM') )
            lumivalidation.setNotNullConstraint('FLAG',True)
            db.createTable(lumivalidation,withIdTable=False)
            created.append(nameDealer.lumivalidationTableName())
        return created
    except :
        raise 

   
#=======================================================
#
#   DROP
#
#=======================================================    
def dropTables(schema,tablelist):
    try:
        db=dbUtil.dbUtil(schema)
        for tablename in tablelist:
            if tablename in [nameDealer.luminormTableName(),nameDealer.lumidataTableName(),nameDealer.trgdataTableName(),nameDealer.hltdataTableName()]:
                db.dropTable( nameDealer.idTableName(tablename) )
                db.dropTable( nameDealer.entryTableName(tablename) )
                db.dropTable( nameDealer.revmapTableName(tablename) )            
            if tablename in [nameDealer.trgTableName(),nameDealer.lumisummaryTableName(),nameDealer.lumisummaryv2TableName(),nameDealer.lumidetailTableName(),nameDealer.hltTableName()]:
                db.dropTable( nameDealer.idTableName(tablename) )
            db.dropTable( tablename )
    except :
        raise 
    
def createOldSchema(schema):
    '''
    create tables of lumidb1 if not exist
    '''
    try:
        created=[]
        db=dbUtil.dbUtil(schema)
        if not schema.existsTable(nameDealer.lumivalidationTableName()):
            lumivalidation=coral.TableDescription()
            lumivalidation.setName( nameDealer.lumivalidationTableName() )
            lumivalidation.insertColumn( 'RUNNUM','unsigned int' )
            lumivalidation.insertColumn( 'CMSLSNUM','unsigned int' )
            lumivalidation.insertColumn( 'FLAG','string' )
            lumivalidation.insertColumn( 'COMMENT','string' )
            lumivalidation.setPrimaryKey(('RUNNUM','CMSLSNUM'))
            lumivalidation.setNotNullConstraint('FLAG',True)
            db.createTable(lumivalidation,withIdTable=False)
            created.append(nameDealer.lumivalidationTableName())
            
        if not schema.existsTable(nameDealer.cmsrunsummaryTableName()):
            cmsrunsummary=coral.TableDescription()
            cmsrunsummary.setName( nameDealer.cmsrunsummaryTableName() )
            cmsrunsummary.insertColumn('RUNNUM','unsigned int')
            cmsrunsummary.insertColumn('HLTKEY','string')
            cmsrunsummary.insertColumn('FILLNUM','unsigned int')
            cmsrunsummary.insertColumn('SEQUENCE','string')
            cmsrunsummary.insertColumn('STARTTIME','time stamp',6)
            cmsrunsummary.insertColumn('STOPTIME','time stamp',6)
            cmsrunsummary.setPrimaryKey('RUNNUM')
            cmsrunsummary.setNotNullConstraint('HLTKEY',True)
            cmsrunsummary.setNotNullConstraint('FILLNUM',True)
            cmsrunsummary.setNotNullConstraint('SEQUENCE',True)
            cmsrunsummary.createIndex('cmsrunsummary_fillnum',('FILLNUM'))
            cmsrunsummary.createIndex('cmsrunsummary_startime',('STARTTIME'))
            db.createTable(cmsrunsummary,withIdTable=False)
            created.append(nameDealer.cmsrunsummaryTableName())
            
        if not schema.existsTable(nameDealer.lumisummaryTableName()):
            summary=coral.TableDescription()
            summary.setName( nameDealer.lumisummaryTableName() )
            summary.insertColumn('LUMISUMMARY_ID','unsigned long long')
            summary.insertColumn('RUNNUM','unsigned int')
            summary.insertColumn('CMSLSNUM','unsigned int')
            summary.insertColumn('LUMILSNUM','unsigned int')
            summary.insertColumn('LUMIVERSION','string')
            summary.insertColumn('DTNORM','float')
            summary.insertColumn('LHCNORM','float')
            summary.insertColumn('CMSALIVE','short')
            summary.insertColumn('INSTLUMI','float')
            summary.insertColumn('INSTLUMIERROR','float')
            summary.insertColumn('INSTLUMIQUALITY','short')
            summary.insertColumn('BEAMSTATUS','string')
            summary.insertColumn('BEAMENERGY','float')
            summary.insertColumn('NUMORBIT','unsigned int')
            summary.insertColumn('STARTORBIT','unsigned int')
            summary.insertColumn('LUMISECTIONQUALITY','short')
            summary.insertColumn('CMSBXINDEXBLOB','blob')
            summary.insertColumn('BEAMINTENSITYBLOB_1','blob')
            summary.insertColumn('BEAMINTENSITYBLOB_2','blob')           
            summary.setPrimaryKey('LUMISUMMARY_ID')
            summary.setNotNullConstraint('RUNNUM',True)
            summary.setNotNullConstraint('CMSLSNUM',True)
            summary.setNotNullConstraint('LUMILSNUM',True)
            summary.setNotNullConstraint('CMSALIVE',True)
            summary.setNotNullConstraint('LUMIVERSION',True)
            summary.setNotNullConstraint('DTNORM',True)
            summary.setNotNullConstraint('LHCNORM',True)
            summary.setNotNullConstraint('INSTLUMI',True)
            summary.setNotNullConstraint('INSTLUMIERROR',True)
            summary.setNotNullConstraint('INSTLUMIQUALITY',True)
            summary.setNotNullConstraint('STARTORBIT',True)
            summary.setNotNullConstraint('NUMORBIT',True)
            summary.setNotNullConstraint('LUMISECTIONQUALITY',True)
            summary.setNotNullConstraint('BEAMENERGY',True)
            summary.setNotNullConstraint('BEAMSTATUS',True)
            summary.setUniqueConstraint(('RUNNUM','LUMIVERSION','LUMILSNUM'))
            summary.createIndex('lumisummary_runnum',('RUNNUM'))
            db.createTable(summary,withIdTable=True)
            created.append(nameDealer.lumisummaryTableName())
            
        if not schema.existsTable(nameDealer.lumidetailTableName()):
            detail=coral.TableDescription()
            detail.setName( nameDealer.lumidetailTableName() )
            detail.insertColumn('LUMIDETAIL_ID','unsigned long long')
            detail.insertColumn('LUMISUMMARY_ID','unsigned long long')
            detail.insertColumn('BXLUMIVALUE','blob')
            detail.insertColumn('BXLUMIERROR','blob')
            detail.insertColumn('BXLUMIQUALITY','blob')
            detail.insertColumn('ALGONAME','string')
            detail.setPrimaryKey('LUMIDETAIL_ID')
            detail.createForeignKey('DETAILSOURCE','LUMISUMMARY_ID',nameDealer.lumisummaryTableName(),'LUMISUMMARY_ID')
            detail.setNotNullConstraint('BXLUMIVALUE',True)
            detail.setNotNullConstraint('BXLUMIERROR',True)
            detail.setNotNullConstraint('BXLUMIQUALITY',True)
            detail.setNotNullConstraint('ALGONAME',True)
            detail.setUniqueConstraint(('LUMISUMMARY_ID','ALGONAME'))
            db.createTable(detail,withIdTable=True)
            created.append(nameDealer.lumidetailTableName())
            
        if  not schema.existsTable(nameDealer.trgTableName()):
            trg=coral.TableDescription()
            trg.setName( nameDealer.trgTableName() )
            trg.insertColumn('TRG_ID','unsigned long long')
            trg.insertColumn('RUNNUM','unsigned int')
            trg.insertColumn('CMSLSNUM','unsigned int')
            trg.insertColumn('BITNUM','unsigned int')
            trg.insertColumn('BITNAME','string')
            trg.insertColumn('TRGCOUNT','unsigned int')
            trg.insertColumn('DEADTIME','unsigned long long')
            trg.insertColumn('PRESCALE','unsigned int')
            trg.setNotNullConstraint('RUNNUM',True)
            trg.setNotNullConstraint('CMSLSNUM',True)
            trg.setNotNullConstraint('BITNUM',True)
            trg.setNotNullConstraint('BITNAME',True)
            trg.setNotNullConstraint('TRGCOUNT',True)
            trg.setNotNullConstraint('DEADTIME',True)
            trg.setNotNullConstraint('PRESCALE',True)
            trg.setPrimaryKey('TRG_ID')
            trg.createIndex('trg_runnum',('RUNNUM'))        
            db.createTable(trg,withIdTable=True)
            created.append( nameDealer.trgTableName() )

        if not schema.existsTable( nameDealer.hltTableName() ): 
            hlt=coral.TableDescription()
            hlt.setName( nameDealer.hltTableName() )
            hlt.insertColumn( 'HLT_ID','unsigned long long')
            hlt.insertColumn( 'RUNNUM','unsigned int')
            hlt.insertColumn( 'CMSLSNUM','unsigned int')
            hlt.insertColumn( 'PATHNAME','string')
            hlt.insertColumn( 'INPUTCOUNT','unsigned int')
            hlt.insertColumn( 'ACCEPTCOUNT','unsigned int')
            hlt.insertColumn( 'PRESCALE','unsigned int')
            hlt.setPrimaryKey( 'HLT_ID' )
            hlt.setNotNullConstraint('RUNNUM',True)
            hlt.setNotNullConstraint('CMSLSNUM',True)
            hlt.setNotNullConstraint('PATHNAME',True)
            hlt.setNotNullConstraint('INPUTCOUNT',True)
            hlt.setNotNullConstraint('ACCEPTCOUNT',True)
            hlt.setNotNullConstraint('PRESCALE',True)
            hlt.createIndex('hlt_runnum',('RUNNUM'))
            db.createTable(hlt,withIdTable=True)
            created.append( nameDealer.hltTableName() )
            
        if not schema.existsTable( nameDealer.trghltMapTableName() ): 
            trghlt=coral.TableDescription()
            trghlt.setName( nameDealer.trghltMapTableName() )
            trghlt.insertColumn( 'HLTKEY','string' )
            trghlt.insertColumn( 'HLTPATHNAME','string' )
            trghlt.insertColumn( 'L1SEED','string' )
            trghlt.setNotNullConstraint('HLTKEY',True)
            trghlt.setNotNullConstraint('HLTPATHNAME',True)
            trghlt.setNotNullConstraint('L1SEED',True)
            db.createTable(trghlt,withIdTable=False)
            created.append( nameDealer.trghltMapTableName() )
        return created
    except:
        raise
        
#=======================================================
#
#   MODIFY
#
#=======================================================
def oldToNew(schema):
    '''
    modify old tables:cmsrunsummary
    alter table cmsrunsummary add column(l1key string,egev unsigned int,amodetag string)
    '''
    try:
        tableHandle=schema.tableHandle(nameDealer.cmsrunsummaryTableName())
        tableHandle.schemaEditor().insertColumn('L1KEY','string')
        tableHandle.schemaEditor().insertColumn('EGEV','unsigned int')
        tableHandle.schemaEditor().insertColumn('AMODETAG','string')
        createTables(schema)
    except:
        raise
    
def newToOld(schema):
    try:
        dropTables(schema,['REVISIONS','LUMINORMS','LUMIDATA','LUMISUMMARYV2','TRGDATA','LSTRG','HLTDATA','LSHLT'])
        tableHandle=schema.tableHandle(nameDealer.cmsrunsummaryTableName())
        ncol=tableHandle.description().numberOfColumns()
        todrop=[]
        for i in range(ncol):
            colname=tableHandle.description().columnDescription(i).name()
            if colname in ['L1KEY','EGEV','AMODETAG']:
                todrop.append(colname)
        for colname in todrop:
            tableHandle.schemaEditor().dropColumn(colname)            
    except :
        raise 

#=======================================================
#
#   TODO
#
#=======================================================   

def createIndices(schema):
    '''
    '''
    pass
    #cmsrunsummary.createIndex('cmsrunsummary_fillnum',('FILLNUM'))
    #cmsrunsummary.createIndex('cmsrunsummary_startime',('STARTTIME'))

def dropIndices(schema):
    '''
    '''
    pass
def describeIndices(schema):
    '''
    '''
    pass
def createFKConstraints(schema):
    '''
    '''
    pass
def dropFKConstrains(schema):
    '''
    '''
    pass
def createNULLConstraints(schema):
    '''
    '''
    #cmsrunsummary.setNotNullConstraint('HLTKEY',True)
    #cmsrunsummary.setNotNullConstraint('FILLNUM',True)
    #cmsrunsummary.setNotNullConstraint('SEQUENCE',True)
    pass
def dropNULLConstraints(schema):
    '''
    '''
    pass

def createUniqueConstraints(schema):
    '''
    '''
    try:
        revtable=schema.tableHandle(nameDealer.revisionTableName())
        revtable.schemaEditor().setUniqueConstraint('NAME','revisions_name_uc')
    except:
        raise
def dropUNIQUEConstraints(schema):
    '''
    '''
    pass

def describe(schema):
    '''
    '''
    pass

if __name__ == "__main__":
    import sessionManager
    myconstr='oracle://devdb10/cms_xiezhen_dev'
    authpath='/afs/cern.ch/user/x/xiezhen'
    #myconstr='sqlite_file:test.db'
    svc=sessionManager.sessionManager(myconstr,authpath=authpath,debugON=False)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    schema=session.nominalSchema()
    session.transaction().start(False)
    tables=createTables(schema)
    if len(tables)==0:
        dropTables(schema,nameDealer.schemaV2Tables())
    else:
        dropTables(schema,tables)
        dropTables(schema,nameDealer.commonTables())
    #createUniqueConstraints(schema)
    #session.transaction().commit()
   
    #print 'droped new '

    session.transaction().start(False)
    tables=createOldSchema(schema)
    oldToNew(schema)
    print 'created old ',tables
    session.transaction().commit()
    del session

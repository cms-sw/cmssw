#!/usr/bin/env python
VERSION='2.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,dbUtil,nameDealer

def createLumi(dbsession):
    print 'creating lumi db schema...'
    dbsession.transaction().start(False)
    schema=dbsession.nominalSchema()
    db=dbUtil.dbUtil(schema)
    #cms run summary table
    
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
    db.createTable(cmsrunsummary,False)

    #lumi summary table
    summary=coral.TableDescription()
    summary.setName( nameDealer.lumisummaryTableName() )
    summary.insertColumn('LUMISUMMARY_ID','unsigned long long')
    summary.insertColumn('RUNNUM','unsigned int')
    summary.insertColumn('CMSLSNUM','unsigned int')
    summary.insertColumn('LUMILSNUM','unsigned int')
    summary.insertColumn('LUMIVERSION','string')
    summary.insertColumn('DTNORM','float')
    summary.insertColumn('LHCNORM','float')
    summary.insertColumn('INSTLUMI','float')
    summary.insertColumn('INSTLUMIERROR','float')
    summary.insertColumn('INSTLUMIQUALITY','short')
    summary.insertColumn('CMSALIVE','short')
    summary.insertColumn('STARTORBIT','unsigned int')
    summary.insertColumn('NUMORBIT','unsigned int')
    summary.insertColumn('LUMISECTIONQUALITY','short')
    summary.insertColumn('BEAMENERGY','float')
    summary.insertColumn('BEAMSTATUS','string')
    summary.insertColumn('CMSBXINDEXBLOB','blob')
    summary.insertColumn('BEAMINTENSITYBLOB_1','blob')
    summary.insertColumn('BEAMINTENSITYBLOB_2','blob')
         
    summary.setPrimaryKey('LUMISUMMARY_ID')
    summary.setNotNullConstraint('RUNNUM',True)
    summary.setNotNullConstraint('CMSLSNUM',True)
    summary.setNotNullConstraint('LUMILSNUM',True)
    summary.setNotNullConstraint('LUMIVERSION',True)
    summary.setNotNullConstraint('DTNORM',True)
    summary.setNotNullConstraint('LHCNORM',True)
    summary.setNotNullConstraint('INSTLUMI',True)
    summary.setNotNullConstraint('INSTLUMIERROR',True)
    summary.setNotNullConstraint('INSTLUMIQUALITY',True)
    summary.setNotNullConstraint('CMSALIVE',True)
    summary.setNotNullConstraint('STARTORBIT',True)
    summary.setNotNullConstraint('NUMORBIT',True)
    summary.setNotNullConstraint('LUMISECTIONQUALITY',True)
    summary.setNotNullConstraint('BEAMENERGY',True)
    summary.setNotNullConstraint('BEAMSTATUS',True)

    summary.setUniqueConstraint(('RUNNUM','LUMIVERSION','LUMILSNUM'))
    summary.createIndex('lumisummary_runnum',('RUNNUM'))
    
    db.createTable(summary,True)
    #lumi detail table
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

    db.createTable(detail,True)
    #trg table
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
    
    db.createTable(trg,True)
    #hlt table
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
    db.createTable(hlt,True)
    #trghlt map table
    trghlt=coral.TableDescription()
    trghlt.setName( nameDealer.trghltMapTableName() )
    #trghlt.insertColumn( 'RUNNUM','unsigned int' )
    trghlt.insertColumn( 'HLTKEY','string' )
    trghlt.insertColumn( 'HLTPATHNAME','string' )
    trghlt.insertColumn( 'L1SEED','string' )
    trghlt.setNotNullConstraint('HLTKEY',True)
    trghlt.setNotNullConstraint('HLTPATHNAME',True)
    trghlt.setNotNullConstraint('L1SEED',True)
    db.createTable(trghlt,False)
    #lumiresult table
    lumiresult=coral.TableDescription()
    lumiresult.setName( nameDealer.lumiresultTableName() )
    lumiresult.insertColumn( 'RUNNUM','unsigned int' )
    lumiresult.insertColumn( 'LUMIVERSION','string' )
    lumiresult.insertColumn( 'DELIVEREDLUMI','float' )
    lumiresult.insertColumn( 'RECORDEDLUMI','float' )
    db.createTable(lumiresult,False)
    #lumihltresult table
    lumihltresult=coral.TableDescription()
    lumihltresult.setName( nameDealer.lumihltresultTableName() )
    lumihltresult.insertColumn( 'RUNNUM','unsigned int' )
    lumihltresult.insertColumn( 'LUMIVERSION','string' )
    lumihltresult.insertColumn( 'HLTPATH','float' )
    lumihltresult.insertColumn( 'RECORDEDLUMI','float' )
    db.createTable(lumihltresult,False)
    
    #lumivalidation table
    lumivalidation=coral.TableDescription()
    lumivalidation.setName( nameDealer.lumivalidationTableName() )
    lumivalidation.insertColumn( 'RUNNUM','unsigned int' )
    lumivalidation.insertColumn( 'CMSLSNUM','unsigned int' )
    lumivalidation.insertColumn( 'FLAG','string' )
    lumivalidation.insertColumn( 'COMMENT','string' )
    lumivalidation.setPrimaryKey(('RUNNUM','CMSLSNUM'))
    lumivalidation.setNotNullConstraint('FLAG',True)
    
    db.createTable(lumivalidation,False)
    dbsession.transaction().commit()
    
def createValidation(dbsession):
    '''
    lumivalidation table
    '''
    dbsession.transaction().start(False)
    schema=dbsession.nominalSchema()
    db=dbUtil.dbUtil(schema)
    lumivalidation=coral.TableDescription()
    lumivalidation.setName( nameDealer.lumivalidationTableName() )
    lumivalidation.insertColumn( 'RUNNUM','unsigned int' )
    lumivalidation.insertColumn( 'CMSLSNUM','unsigned int' )
    lumivalidation.insertColumn( 'FLAG','string' )
    lumivalidation.insertColumn( 'COMMENT','string' )
    lumivalidation.setPrimaryKey(('RUNNUM','CMSLSNUM'))
    lumivalidation.setNotNullConstraint('FLAG',True)
    db.createTable(lumivalidation,False)
    dbsession.transaction().commit()
    
def dropLumi(dbsession):
    print 'droping lumi db schema...'
    dbsession.transaction().start(False)
    schema=dbsession.nominalSchema()
    db=dbUtil.dbUtil(schema)
    db.dropTable( nameDealer.lumidetailTableName() )
    db.dropTable( nameDealer.cmsrunsummaryTableName() )
    db.dropTable( nameDealer.lumisummaryTableName() )
    db.dropTable( nameDealer.trgTableName() )
    db.dropTable( nameDealer.hltTableName() )
    db.dropTable( nameDealer.trghltMapTableName() )
    db.dropTable( nameDealer.lumiresultTableName() )
    db.dropTable( nameDealer.lumihltresultTableName() )
    db.dropTable( nameDealer.lumivalidationTableName() )
    dbsession.transaction().commit()
    
def describeLumi(dbsession):
    print 'lumi db schema dump...'
    dbsession.transaction().start(True)
    schema=dbsession.nominalSchema()
    db=dbUtil.dbUtil(schema)
    db.describeSchema()
    dbsession.transaction().commit()

def createIndex(dbsession):
    dbsession.transaction().start(False)
    schema=dbsession.nominalSchema()
    schema.tableHandle( nameDealer.lumisummaryTableName() ).schemaEditor().createIndex('lumisummary_runnum',('RUNNUM'))
    schema.tableHandle( nameDealer.trgTableName() ).schemaEditor().createIndex('trg_runnum',('RUNNUM'))
    schema.tableHandle( nameDealer.hltTableName() ).schemaEditor().createIndex('hlt_runnum',('RUNNUM'))
    dbsession.transaction().commit()
    
def dropIndex(dbsession):
    dbsession.transaction().start(False)
    schema=dbsession.nominalSchema()
    schema.tableHandle( nameDealer.lumisummaryTableName() ).schemaEditor().dropIndex('lumisummary_runnum')
    schema.tableHandle( nameDealer.trgTableName() ).schemaEditor().dropIndex('trg_runnum')
    schema.tableHandle( nameDealer.hltTableName() ).schemaEditor().dropIndex('hlt_runnum')
    dbsession.transaction().commit()
    
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi DB schema operations.")
    # add the arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('action',choices=['create','drop','describe','addindex','dropindex'],help='action on the schema')
    parser.add_argument('--validationTab',dest='validationTab',action='store_true',help='validation table only')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug mode')
    # parse arguments
    args=parser.parse_args()
    connectstring=args.connect
    if args.debug:
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    svc = coral.ConnectionService()
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    session=svc.connect(connectstring,accessMode=coral.access_Update)
    if args.action == 'create':
       if args.validationTab:
           createValidation(session)
       else:
           createLumi(session)
    if args.action == 'drop':
       dropLumi(session)
    if args.action == 'describe':
       describeLumi(session)
    if args.action == 'addindex':
       createIndex(session)
    if args.action == 'dropindex':
       dropIndex(session)
    if args.verbose :
        print 'verbose mode'
if __name__=='__main__':
    main()
    

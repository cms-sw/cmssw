#!/usr/bin/env python
VERSION='2.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,dbUtil,nameDealer,lumidbDDL,dataDML,revisionDML

def createLumi(dbsession):
    print 'creating lumidb2 schema...'
    dbsession.transaction().start(False)
    schema=dbsession.nominalSchema()
    lumidbDDL.createTables(schema)
    dbsession.transaction().commit()
    
def dropLumi(dbsession):
    print 'droping lumi db2 schema...'
    dbsession.transaction().start(False)
    schema=dbsession.nominalSchema()
    lumidbDDL.dropTables(schema,nameDealer.schemaV2Tables())
    dbsession.transaction().commit()
    
def describeLumi(dbsession):
    print 'lumi db schema dump...'
    dbsession.transaction().start(True)
    schema=dbsession.nominalSchema()
    db=dbUtil.dbUtil(schema)
    db.describeSchema()
    dbsession.transaction().commit()

def createIndex(dbsession):
    pass
    
def dropIndex(dbsession):
    pass

def createBranch(dbsession,branchname,parentname,comment):
    print 'creating branch ',branchname
    dbsession.transaction().start(False)
    (branchid,parentid,parentname)=revisionDML.createBranch(dbsession.nominalSchema(),branchname,parentname,comment)
    dbsession.transaction().commit()
    print 'branchid ',branchid,' parentname ',parentname,' parentid ',parentid
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi DB schema operations.")
    # add the arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('action',choices=['create','drop','describe','addindex','dropindex'],help='action on the schema')
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
        createLumi(session)
        createBranch(session,'TRUNK',None,'root')
        createBranch(session,'NORM','TRUNK','hold normalization factor')
        createBranch(session,'DATA','TRUNK','hold data')
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
    

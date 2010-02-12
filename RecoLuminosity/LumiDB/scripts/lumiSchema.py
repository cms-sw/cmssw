#!/usr/bin/env python
VERSION='1.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,dbUtil,nameDealer

def createLumi(dbsession):
    print 'creating lumi db schema...'
    dbsession.transaction().start(False)
    schema=dbsession.nominalSchema()
    description=coral.TableDescription()
    description.setName( nameDealer.lumisummaryTableName() )
    description.insertColumn('LUMISUMMARY_ID','unsigned long long')
    description.insertColumn('RUNNUM','unsigned long')
    description.insertColumn('LUMILSNUM','unsigned long')
    description.insertColumn('LUMIVERSION','string')
    description.insertColumn('DTNORM','float')
    description.insertColumn('LUMINORM','float')
    description.insertColumn('INSTLUMI','float')
    description.insertColumn('INSTLUMIERROR','float')
    description.insertColumn('INSTLUMIQUALITY','short')
    description.insertColumn('CMSALIVE','bool')
    description.insertColumn('LUMISECTIONQUALITY','short')
    description.setPrimaryKey('LUMISUMMARY_ID')
    db=dbUtil.dbUtil(schema)
    db.createTable(description,False)    
    dbsession.transaction().commit()

def dropLumi(dbsession):
    print 'droping lumi db schema...'
    
def describeLumi(dbsession):
    print 'lumi db schema'
    dbsession.transaction().start(True)
    schema=dbsession.nominalSchema()
    db=dbUtil.dbUtil(schema)
    db.describeSchema()
    dbsession.transaction().commit()
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi DB schema operations.")
    # add the arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')    
    parser.add_argument('action',choices=['create','drop','describe'],help='action on the schema')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose')
    # parse arguments
    args=parser.parse_args()
    connectstring=args.connect
    svc = coral.ConnectionService()
    session=svc.connect(connectstring,accessMode=coral.access_Update)
    if args.action == 'create':
       createLumi(session)
    if args.action == 'drop':
       dropLumi(session)
    if args.action == 'describe':
       describeLumi(session) 
    if args.verbose :
        print 'verbose mode'
if __name__=='__main__':
    main()
    

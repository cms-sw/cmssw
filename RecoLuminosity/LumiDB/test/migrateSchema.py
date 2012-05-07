#!/usr/bin/env python
VERSION='2.00'
import os,sys,array
import coral
from RecoLuminosity.LumiDB import argparse,idDealer,nameDealer,CommonUtil,lumidbDDL,dbUtil,dataDML,revisionDML
#
# migrate lumidb v1 data to v2 schema, complete missing runsummary fields
# data transfer section
#
# can be used for data copy program
#
DATABRANCH_ID=3
def main():
    from RecoLuminosity.LumiDB import sessionManager,queryDataSource    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="migrate lumidb schema",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',dest='connect',action='store',required=False,default='oracle://devdb10/cms_xiezhen_dev',help='connect string to dest db(required)')
    parser.add_argument('-lumisource',dest='lumisource',action='store',required=False,default='oracle://cms_orcoff_prod/CMS_LUMI_PROD',help='connect string to source lumi db')
    parser.add_argument('-runinfo',dest='runinfo',action='store',required=False,default='oracle://cms_orcoff_prod/CMS_RUNINFO',help='connect string to runinfo db')
    parser.add_argument('-P',dest='authpath',action='store',required=False,default='/afs/cern.ch/user/x/xiezhen',help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',required=True,help='run number')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=int(args.runnumber)
    print 'processing run ',runnumber
    runinfosvc=sessionManager.sessionManager(args.runinfo,authpath=args.authpath,debugON=args.debug)
    lumisvc=sessionManager.sessionManager(args.lumisource,authpath=args.authpath,debugON=args.debug)
    destsvc=sessionManager.sessionManager(args.connect,authpath=args.authpath,debugON=args.debug)
    print 'fetch runsummary from runinfo'
    
    runinfosession=runinfosvc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    
    lumisession=lumisvc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    try:
        [l1key,amodetag,egev,sequence,hltkey,fillnum,starttime,stoptime]=queryDataSource.runsummary(runinfosession,'CMS_RUNINFO',runnumber,complementalOnly=False)
        print 'runsummary ',[l1key,amodetag,egev,sequence,hltkey,fillnum,starttime,stoptime]
        lumidata=queryDataSource.uncalibratedlumiFromOldLumi(lumisession,runnumber)
       #print 'lumidata ',lumidata
        [bitnames,trglsdata]=queryDataSource.trgFromOldLumi(lumisession,runnumber)
       #print 'trg data ',bitnames,trglsdata 
        [pathnames,hltlsdata]=queryDataSource.hltFromOldLumi(lumisession,runnumber)
       #print 'hlt data ',pathnames,hltlsdata
        lumisession.transaction().commit()
        runinfosession.transaction().commit()
        del lumisession
        del lumisvc
        del runinfosession
        del runinfosvc
        destsession=destsvc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        destsession.transaction().start(False)
        branchrevision_id=DATABRANCH_ID
        #print 'data branchid ',branchrevision_id
        dataDML.insertRunSummaryData(destsession.nominalSchema(),runnumber,[l1key,amodetag,egev,sequence,hltkey,fillnum,starttime,stoptime],complementalOnly=False)
        (lumirevid,lumientryid,lumidataid)=dataDML.addLumiRunDataToBranch(destsession.nominalSchema(),runnumber,[args.lumisource],(branchrevision_id,'DATA'))
        bitzeroname=bitnames.split(',')[0]
        trgrundata=[args.lumisource,bitzeroname,bitnames]
        (trgrevid,trgentryid,trgdataid)=dataDML.addTrgRunDataToBranch(destsession.nominalSchema(),runnumber,trgrundata,(branchrevision_id,'DATA'))
        hltrundata=[pathnames,args.lumisource]
        (hltrevid,hltentryid,hltdataid)=dataDML.addHLTRunDataToBranch(destsession.nominalSchema(),runnumber,hltrundata,(branchrevision_id,'DATA'))
        destsession.transaction().commit()
        dataDML.bulkInsertLumiLSSummary(destsession,runnumber,lumidataid,lumidata,500)
        #
        dataDML.bulkInsertTrgLSData(destsession,runnumber,trgdataid,trglsdata,500)
        dataDML.bulkInsertHltLSData(destsession,runnumber,hltdataid,hltlsdata,500)
        del destsession
        del destsvc
    except:
        raise
        
if __name__=='__main__':
    main()
    

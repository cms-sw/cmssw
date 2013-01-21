#!/usr/bin/env python
import os,sys,time,csv,coral
from RecoLuminosity.LumiDB import sessionManager,argparse,nameDealer,revisionDML,dataDML,lumiParameters

def generateLumiLSdataForRun(lsdatafromfile,rundatafromfile,lsdatafromdb,rundatafromdb):
    '''
    input:
      lsdata: [(cmslsnum,instlumi),...]
      lumirundata: [datasource,nominalegev,ncollidingbunches]
      beamstatus {cmslsnum:beamstatus}
    output:
    i.e. bulkInsertLumiLSSummary expected input: {lumilsnum:[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit]}
    '''
    lumip=lumiParameters.ParametersObject()
    result={}
    beamstatus='STABLE BEAMS'
    beamenergy=lumirundata[1]
    numorbit=lumip.numorbit
    startorbit=0
    for (cmslsnum,instlumi) in lsdata:
        lumilsnum=cmslsnum
        instlumierror=0.0
        instlumiquality=0
        startorbit=(cmslsnum-1)*numorbit
        if beamsta and beamsta.has_key(cmslsnum):
            beamstatus=beamsta[cmslsnum]
        result[lumilsnum]=[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit]
    return result

def lumiDataFromfile(filename):
    '''
    input:bcm1f lumi csv file
    output:{runnumber,[(lumilsnum,instlumi)]}
    '''
    perrunresult=[]#source,starttime,stoptime
    perlsresult={}#{lumilsnum:instlumiub}
    csv_data=open(filename)
    
    csvreader=csv.reader(csv_data,delimiter=',')
    idx=0
    ts=[]
    for row in csvreader:
        if idx==0:
            idx=1
            continue
        if row[0].find('#')==1:
            continue
        if not row:
            continue
        tsunix=int(row[0].strip())
        ts.append(tsunix)
        lumils=int(row[1].strip())
        instdelivered=float(row[2].strip())/23.31/1.0e3 #convert to instlumi
        perlsresult[lumils]=instdelivered
        idx+=1
    startts=min(ts)
    stopts=max(ts)
    nls=len(perlsresult)
    perrunresult=[filename,startts,stopts,nls]
    return (perrunresult,perlsresult)

def lumiDataFromDB(sourceschema,sourcelumidataid):
    '''
    select nominalegev,ncollidingbunches from lumidata where DATA_ID=:dataid
    select lumilsnum,cmslsnum,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexblob,beamintensityblob_1,beamintensityblob_2,bxlumivalue_occ1,bxlumivalue_occ2,bxlumivalue_et,bxlumierror_occ1,bxlumierror_occ2,bxlumierror_et from lumisummaryv2 where data_id=:dataid
    output:
           (lumirundata,lumilsdata)
           lumirundata=[nominalegev,ncollidingbunches]
           lumilsdata={}#{lumilsnum:[cmslsnum,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexblob,beamintensityblob_1,beamintensityblob_2,bxlumivalue_occ1,bxlumivalue_occ2,bxlumivalue_et,bxlumierror_occ1,bxlumierror_occ2,bxlumierror_et]}
    '''
    lumirundata=[]
    lumilsdata={}
    qHandle=sourceschema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.lumidataTableName() )
        qHandle.addToOutputList('NOMINALEGEV')
        qHandle.addToOutputList('NCOLLIDINGBUNCHES')
        qCondition=coral.AttributeList()
        qCondition.extend('lumidataid','unsigned long long')
        qCondition['lumidataid'].setData(sourcelumidataid)
        qResult=coral.AttributeList()
        qResult.extend('NOMINALEGEV','float')
        qResult.extend('NCOLLIDINGBUNCHES','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:lumidataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            nominalegev=cursor.currentRow()['NOMINALEGEV'].data()
            ncollidingbunches=cursor.currentRow()['NCOLLIDINGBUNCHES'].data()
            lumirundata=[nominalegev,ncollidingbunches]
        del qHandle
    except:
        if qHandle:del qHandle
        raise
    qHandle=sourceschema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.lumisummaryv2TableName() )
        qHandle.addToOutputList('LUMILSNUM')
        qHandle.addToOutputList('CMSLSNUM')
        qHandle.addToOutputList('BEAMSTATUS')
        qHandle.addToOutputList('BEAMENERGY')
        qHandle.addToOutputList('NUMORBIT')
        qHandle.addToOutputList('STARTORBIT')
        qHandle.addToOutputList('CMSBXINDEXBLOB')
        qHandle.addToOutputList('BEAMINTENSITYBLOB_1')
        qHandle.addToOutputList('BEAMINTENSITYBLOB_2')
        qHandle.addToOutputList('BXLUMIVALUE_OCC1')
        qHandle.addToOutputList('BXLUMIVALUE_OCC2')
        qHandle.addToOutputList('BXLUMIVALUE_ET')
        qHandle.addToOutputList('BXLUMIERROR_OCC1')
        qHandle.addToOutputList('BXLUMIERROR_OCC2')
        qHandle.addToOutputList('BXLUMIERROR_ET')
        
        qCondition=coral.AttributeList()
        qCondition.extend('lumidataid','unsigned long long')
        qCondition['lumidataid'].setData(sourcelumidataid)
        qResult=coral.AttributeList()
        qResult.extend('LUMILSNUM','unsigned int')
        qResult.extend('CMSLSNUM','unsigned int')
        qResult.extend('BEAMSTATUS','string')
        qResult.extend('BEAMENERGY','float')
        qResult.extend('NUMORBIT','unsigned int')
        qResult.extend('STARTORBIT','unsigned int')
        qResult.extend('CMSBXINDEXBLOB','blob')
        qResult.extend('BEAMINTENSITYBLOB_1','blob')
        qResult.extend('BEAMINTENSITYBLOB_2','blob')
        qResult.extend('BXLUMIVALUE_OCC1','blob')
        qResult.extend('BXLUMIVALUE_OCC2','blob')
        qResult.extend('BXLUMIVALUE_ET','blob')
        qResult.extend('BXLUMIERROR_OCC1','blob')
        qResult.extend('BXLUMIERROR_OCC2','blob')
        qResult.extend('BXLUMIERROR_ET','blob')
        
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:lumidataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            lumilsnum=cursor.currentRow()['LUMILSNUM'].data()
            cmslsnum=cursor.currentRow()['CMSLSNUM'].data()
            beamstatus=cursor.currentRow()['BEAMSTATUS'].data()
            beamenergy=cursor.currentRow()['BEAMENERGY'].data()
            numorbit=cursor.currentRow()['NUMORBIT'].data()
            startorbit=cursor.currentRow()['STARTORBIT'].data()
            cmsbxindexblob=None
            if not cursor.currentRow()['CMSBXINDEXBLOB'].isNull():
                cmsbxindexblob=cursor.currentRow()['CMSBXINDEXBLOB'].data()
            beamintensityblob_1=None
            if not cursor.currentRow()['BEAMINTENSITYBLOB_1'].isNull():
                beamintensityblob_1=cursor.currentRow()['BEAMINTENSITYBLOB_1'].data()
            beamintensityblob_2=None
            if not cursor.currentRow()['BEAMINTENSITYBLOB_2'].isNull():
                beamintensityblob_1=cursor.currentRow()['BEAMINTENSITYBLOB_2'].data()
            bxlumivalue_occ1=None
            bxlumivalue_occ2=None
            bxlumivalue_et=None
            bxlumierror_occ1=None
            bxlumierror_occ2=None
            bxlumierror_et=None
            lumilsdata[lumilsnum]=[cmslsnum,beamstatus,numorbit,startorbit,cmsbxindexblob,beamintensityblob_1,beamintensityblob_2,bxlumivalue_occ1,bxlumivalue_occ2,bxlumivalue_et,bxlumierror_occ1,bxlumierror_occ2,bxlumierror_et]
        del qHandle
    except:
        if qHandle:del qHandle
        raise
    print lumirundata
    print len(lumilsdata)
    return (lumirundata,lumilsdata)
##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "csv lumi loader",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s',dest='sourcestr',action='store',
                        required=True,
                        help='connect string to source DB (required)',
                        )
    parser.add_argument('-d',dest='deststr',action='store',
                        required=True,
                        help='connect string to dest DB (required)',
                        )
    parser.add_argument('-P',dest='authpath',action='store',
                        required=True,
                        help='path to authentication file (required)'
                        )
    parser.add_argument('-i',dest='inputfile',action='store',
                        required=True,
                        help='input file'
                        )
    parser.add_argument('-r',dest='runnum',action='store',
                        required=True,
                        help='run'
                        )
    parser.add_argument('--comment',action='store',
                        required=False,
                        help='patch comment'
                       )
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug'
                        )
    options=parser.parse_args()

    inputfilename=os.path.abspath(options.inputfile)
    lumidatafromfile=lumiDataFromfile(inputfilename)
    sourcesvc=sessionManager.sessionManager(options.sourcestr,authpath=options.authpath,debugON=False)
    sourcesession=sourcesvc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    sourcesession.transaction().start(True)
    qHandle=sourcesession.nominalSchema().newQuery()
    sourcetagid=0
    sourcelumiid=0
    try:
        qHandle.addToTableList( nameDealer.tagRunsTableName() )
        qHandle.addToOutputList('TAGID')
        qHandle.addToOutputList('LUMIDATAID')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(options.runnum))
        qResult=coral.AttributeList()
        qResult.extend('TAGID','unsigned long long')
        qResult.extend('LUMIDATAID','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            tagid=0
            if not cursor.currentRow()['TAGID'].isNull():
                tagid=cursor.currentRow()['TAGID'].data()
                if tagid>sourcetagid:
                    sourcetagid=tagid
                    sourcelumiid=cursor.currentRow()['LUMIDATAID'].data()
        del qHandle
    except:
        if qHandle:del qHandle
        raise
    print sourcetagid,sourcelumiid
    lumidatafromdb=lumiDataFromDB(sourcesession.nominalSchema(),sourcelumiid)
    sourcesession.transaction().commit()
    del sourcesession
    del sourcesvc

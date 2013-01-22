#!/usr/bin/env python
import os,sys,time,csv,coral
from RecoLuminosity.LumiDB import sessionManager,argparse,nameDealer,revisionDML,dataDML,lumiParameters


def generateLumidata(lumirundatafromfile,lsdatafromfile,rundatafromdb,lsdatafromdb,replacelsMin,replacelsMax):
    '''
    input:
     perrunresultfromfile=[]#source,starttime,stoptime,nls
     perlsresultfromfile={} #{lumilsnum:instlumiub}
     lumirundatafromdb=[]   #[source,nominalegev,ncollidingbunches,starttime,stoptime,nls]
     lumilsdatafromdb={}#{lumilsnum:[cmslsnum(0),instlumi(1),instlumierror(2),instlumiquality(3),beamstatus(4),beamenergy(5),numorbit(6),startorbit(7),cmsbxindexblob(8),beamintensityblob_1(9),beamintensityblob_2(10),bxlumivalue_occ1(11),bxlumierror_occ1(12),bxlumiquality_occ1(13),bxlumivalue_occ2(14),bxlumierror_occ2(15),bxlumiquality_occ2(16),bxlumivalue_et(17),bxlumierror_et(18),bxlumiquality_et(19)]}


    '''
    lumirundata=[]
    lumilsdata=lsdatafromdb
    rundatafromdb[0]=rundatafromdb[0]+'+file:'+lumirundatafromfile[0]
    lumirundata=rundatafromdb
    for lumilsnum in lsdatafromdb.keys():
        if lumilsnum in range(replacelsMin,replacelsMax+1):
            instlumi=lsdatafromfile[lumilsnum]
            lumilsdata[lumilsnum][2]=instlumi
    return (lumirundata,lumilsdata)

def lumiDataFromfile(filename):
    '''
    input:bcm1f lumi csv file
    output:(perrunresult,perlsresult)
            perrunresult=[]#source,starttime,stoptime
            perlsresult={}#{lumilsnum:instlumiub}
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
    select nominalegev,ncollidingbunches,starttime,stoptime,nls from lumidata where DATA_ID=:dataid
    select lumilsnum,cmslsnum,beamstatus,beamenergy,numorbit,startorbit,instlumi,cmsbxindexblob,beamintensityblob_1,beamintensityblob_2,bxlumivalue_occ1,bxlumivalue_occ2,bxlumivalue_et,bxlumierror_occ1,bxlumierror_occ2,bxlumierror_et from lumisummaryv2 where data_id=:dataid
    output:(lumirundata,lumilsdata)
           lumirundata=[source,nominalegev,ncollidingbunches,starttime,stoptime,nls]
           lumilsdata={}#{lumilsnum:[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexblob,beam1intensity,beam2intensity,bxlumivalue_occ1,bxlumierror_occ1,bxlumiquality_occ1,bxlumivalue_occ2,bxlumierror_occ2,bxlumiquality_occ2,bxlumivalue_et,bxlumierror_et,bxlumiquality_et]
    '''
    lumirundata=[]
    lumilsdata={}
    qHandle=sourceschema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.lumidataTableName() )
        qHandle.addToOutputList('SOURCE')
        qHandle.addToOutputList('NOMINALEGEV')
        qHandle.addToOutputList('NCOLLIDINGBUNCHES')
        #qHandle.addToOutputList('TO_CHAR(STARTTIME,\'MM/DD/YY HH24:MI:SS\')','starttime')
        #qHandle.addToOutputList('TO_CHAR(STOPTIME,\'MM/DD/YY HH24:MI:SS\')','stoptime')
        qHandle.addToOutputList('STARTTIME')
        qHandle.addToOutputList('STOPTIME')
        qHandle.addToOutputList('NLS')
        qCondition=coral.AttributeList()
        qCondition.extend('lumidataid','unsigned long long')
        qCondition['lumidataid'].setData(sourcelumidataid)
        qResult=coral.AttributeList()
        qResult.extend('SOURCE','string')
        qResult.extend('NOMINALEGEV','float')
        qResult.extend('NCOLLIDINGBUNCHES','unsigned int')
        #qResult.extend('starttime','string')
        #qResult.extend('stoptime','string')
        qResult.extend('STARTTIME','time stamp')
        qResult.extend('STOPTIME','time stamp')
        qResult.extend('NLS','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:lumidataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            source=cursor.currentRow()['SOURCE'].data()
            nominalegev=cursor.currentRow()['NOMINALEGEV'].data()
            ncollidingbunches=cursor.currentRow()['NCOLLIDINGBUNCHES'].data()
            #starttime=cursor.currentRow()['starttime'].data()
            #stoptime=cursor.currentRow()['stop'].data()
            starttime=cursor.currentRow()['STARTTIME'].data()
            stoptime=cursor.currentRow()['STOPTIME'].data()
            nls=cursor.currentRow()['NLS'].data()
            lumirundata=[source,nominalegev,ncollidingbunches,starttime,stoptime,nls]
        del qHandle
    except:
        if qHandle:del qHandle
        raise
    qHandle=sourceschema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.lumisummaryv2TableName() )
        qHandle.addToOutputList('LUMILSNUM')
        qHandle.addToOutputList('CMSLSNUM')
        qHandle.addToOutputList('INSTLUMI')
        qHandle.addToOutputList('INSTLUMIERROR')
        qHandle.addToOutputList('INSTLUMIQUALITY')
        qHandle.addToOutputList('BEAMSTATUS')
        qHandle.addToOutputList('BEAMENERGY')
        qHandle.addToOutputList('NUMORBIT')
        qHandle.addToOutputList('STARTORBIT')
        qHandle.addToOutputList('CMSBXINDEXBLOB')
        qHandle.addToOutputList('BEAMINTENSITYBLOB_1')
        qHandle.addToOutputList('BEAMINTENSITYBLOB_2')
        qHandle.addToOutputList('BXLUMIVALUE_OCC1')
        qHandle.addToOutputList('BXLUMIERROR_OCC1')
        qHandle.addToOutputList('BXLUMIQUALITY_OCC1')
        qHandle.addToOutputList('BXLUMIVALUE_OCC2')
        qHandle.addToOutputList('BXLUMIERROR_OCC2')
        qHandle.addToOutputList('BXLUMIQUALITY_OCC2')
        qHandle.addToOutputList('BXLUMIVALUE_ET')
        qHandle.addToOutputList('BXLUMIERROR_ET')
        qHandle.addToOutputList('BXLUMIQUALITY_ET')
        qCondition=coral.AttributeList()
        qCondition.extend('lumidataid','unsigned long long')
        qCondition['lumidataid'].setData(sourcelumidataid)
        qResult=coral.AttributeList()
        qResult.extend('LUMILSNUM','unsigned int')
        qResult.extend('CMSLSNUM','unsigned int')
        qResult.extend('INSTLUMI','float')
        qResult.extend('INSTLUMIERROR','float')
        qResult.extend('INSTLUMIQUALITY','short')
        qResult.extend('BEAMSTATUS','string')
        qResult.extend('BEAMENERGY','float')
        qResult.extend('NUMORBIT','unsigned int')
        qResult.extend('STARTORBIT','unsigned int')
        qResult.extend('CMSBXINDEXBLOB','blob')
        qResult.extend('BEAMINTENSITYBLOB_1','blob')
        qResult.extend('BEAMINTENSITYBLOB_2','blob')
        qResult.extend('BXLUMIVALUE_OCC1','blob')
        qResult.extend('BXLUMIERROR_OCC1','blob')
        qResult.extend('BXLUMIQUALITY_OCC1','blob')        
        qResult.extend('BXLUMIVALUE_OCC2','blob')
        qResult.extend('BXLUMIERROR_OCC2','blob')
        qResult.extend('BXLUMIQUALITY_OCC2','blob')
        qResult.extend('BXLUMIVALUE_ET','blob')
        qResult.extend('BXLUMIERROR_ET','blob')
        qResult.extend('BXLUMIQUALITY_ET','blob')
        
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:lumidataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            lumilsnum=cursor.currentRow()['LUMILSNUM'].data()
            cmslsnum=cursor.currentRow()['CMSLSNUM'].data()
            instlumi=cursor.currentRow()['INSTLUMI'].data()
            instlumierror=cursor.currentRow()['INSTLUMIERROR'].data()
            instlumiquality=cursor.currentRow()['INSTLUMIQUALITY'].data()
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
                beamintensityblob_2=cursor.currentRow()['BEAMINTENSITYBLOB_2'].data()
            bxlumivalue_occ1=None
            if not cursor.currentRow()['BXLUMIVALUE_OCC1'].isNull():
                bxlumivalue_occ1=cursor.currentRow()['BXLUMIVALUE_OCC1'].data()
            bxlumivalue_occ2=None
            if not cursor.currentRow()['BXLUMIVALUE_OCC2'].isNull():
                bxlumivalue_occ2=cursor.currentRow()['BXLUMIVALUE_OCC2'].data()
            bxlumivalue_et=None
            if not cursor.currentRow()['BXLUMIVALUE_ET'].isNull():
                bxlumivalue_et=cursor.currentRow()['BXLUMIVALUE_ET'].data()
            bxlumierror_occ1=None
            if not cursor.currentRow()['BXLUMIERROR_OCC1'].isNull():
                bxlumierror_occ1=cursor.currentRow()['BXLUMIERROR_OCC1'].data()
            bxlumierror_occ2=None
            if not cursor.currentRow()['BXLUMIERROR_OCC2'].isNull():
                bxlumierror_occ2=cursor.currentRow()['BXLUMIERROR_OCC2'].data()
            bxlumierror_et=None
            if not cursor.currentRow()['BXLUMIERROR_ET'].isNull():
                bxlumierror_et=cursor.currentRow()['BXLUMIERROR_ET'].data()
            bxlumiquality_occ1=None
            if not cursor.currentRow()['BXLUMIQUALITY_OCC1'].isNull():
                bxlumiquality_occ1=cursor.currentRow()['BXLUMIQUALITY_OCC1'].data()
            bxlumiquality_occ2=None
            if not cursor.currentRow()['BXLUMIQUALITY_OCC2'].isNull():
                bxlumiquality_occ2=cursor.currentRow()['BXLUMIQUALITY_OCC2'].data()
            bxlumiquality_et=None
            if not cursor.currentRow()['BXLUMIQUALITY_ET'].isNull():
                bxlumiquality_et=cursor.currentRow()['BXLUMIQUALITY_ET'].data()
            lumilsdata[lumilsnum]=[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexblob,beamintensityblob_1,beamintensityblob_2,bxlumivalue_occ1,bxlumierror_occ1,bxlumiquality_occ1,bxlumivalue_occ2,bxlumierror_occ2,bxlumiquality_occ2,bxlumivalue_et,bxlumierror_et,bxlumiquality_et]
        del qHandle
    except:
        if qHandle:del qHandle
        raise
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
    parser.add_argument('-b',dest='begLS',action='store',
                        required=True,
                        help='begin LS to be replaced with file data'
                        )
    parser.add_argument('-e',dest='endLS',action='store',
                        required=True,
                        help='end LS to be replaced with file data'
                        )
    parser.add_argument('--comment',action='store',
                        required=False,
                        help='patch comment'
                       )
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug'
                        )
    options=parser.parse_args()
    begLS=int(options.begLS)
    endLS=int(options.endLS)
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
    lumidatafromdb=lumiDataFromDB(sourcesession.nominalSchema(),sourcelumiid)
    sourcesession.transaction().commit()
    (rundat,lsdat)=generateLumidata(lumidatafromfile[0],lumidatafromfile[1],lumidatafromdb[0],lumidatafromdb[1],begLS,endLS)
    print rundat
    print lsdat
    destsvc=sessionManager.sessionManager(options.deststr,authpath=options.authpath,debugON=False)
    destsession=sourcesvc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    destsession.transaction().start(True)
    (lumibranchid,lumibranchparent)=revisionDML.branchInfoByName(destsession.nominalSchema(),'DATA')
    (hf_tagid,hf_tagname)=revisionDML.currentDataTag(destsession.nominalSchema(),lumitype='HF')
    print (hf_tagid,hf_tagname)
    hfdataidmap=revisionDML.dataIdsByTagId(destsession.nominalSchema(),hf_tagid,[int(options.runnum)],withcomment=False,lumitype='HF')
    print hfdataidmap
    destsession.transaction().commit()    
    branchinfo=(lumibranchid,'DATA')
    print branchinfo
    [destlumidataid,desttrgdataid,desthltdataid]=hfdataidmap[int(options.runnum)]
    #destsession.transaction().start(False)
    #(lumirevid,lumientryid,lumidataid)=dataDML.addLumiRunDataToBranch(schema,int(options.runnum),rundat,branchinfo,nameDealer.lumidataTableName())
    #dataDML.bulkInsertLumiLSSummary(destsession,int(options.runnum),lumidataid,lsdat,nameDealer.lumisummaryv2TableName())
    #revisionDML.addRunToCurrentDataTag(destsession.nominalSchema(),int(options.runnum),lumidataid,desttrgdataid,desthltdataid,lumitype='HF')
    #destsession.transaction().commit()
    del sourcesession
    del sourcesvc

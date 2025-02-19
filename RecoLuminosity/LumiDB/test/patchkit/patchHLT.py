import sys,os,csv,re,coral,array
from RecoLuminosity.LumiDB import argparse,sessionManager,CommonUtil,dataDML,revisionDML,nameDealer,dbUtil
DATABRANCH_ID=3
def patchV2(dbsession,runnum,inputpathnames,inputdata):
    '''
    inputpathnames: [pathname,]
    inputdata: {cmslsnum:[presc,presc...]}
    update the most recent version of lshlt data if some ls exist
    if none old ls exists
    insert new hlt records
    '''
    try:
        dbsession.transaction().start(True)
        oldhltdataid=dataDML.guessHltDataIdByRun(dbsession.nominalSchema(),runnum)
        existingrundata=dataDML.hltRunById(dbsession.nominalSchema(),oldhltdataid)
        dbsession.transaction().commit()
        if not oldhltdataid:#no data at all
            dbsession.transaction().start(False)
            insertV2(dbsession,runnum,inputpathnames,inputdata)
            dbsession.transaction().commit()
            return
        hltnamedict=existingrundata[3]#[(pathidx,hltname),(pathidx,hltname)...]
        dbsession.transaction().start(True)
        existinglsdata=dataDML.hltLSById(dbsession.nominalSchema(),oldhltdataid)
        dbsession.transaction().commit()
        oldlsdata=existinglsdata[1]
        existinglslist=oldlsdata.keys()
        toupdate={}#{cmslsnum:[presc,presc...]}
        toinsert={}#{cmslsnum:[presc,presc...]}
        if existinglslist and len(existinglslist)!=0:#there are some existing data
            for cmslsnum,oldlscontent in oldlsdata.items():
                if cmslsnum in inputdata.keys(): # if overlap with new data, update old data with new 
                    toupdate[cmslsnum]=inputdata[cmslsnum]
        for cmslsnum,lshltcontent in inputdata.items():
            if toupdate.has_key(cmslsnum): continue #it's to update not to insert
            toinsert[cmslsnum]=inputdata[cmslsnum]
        #
        # insert into lshlt(data_id,runnum,cmslsnum,prescaleblob,hltcountblob,hltacceptblob) values()
        #
        dbsession.transaction().start(False)
        tabrowDefDict={'DATA_ID':'unsigned long long','RUNNUM':'unsigned int','CMSLSNUM':'unsigned int','PRESCALEBLOB':'blob','HLTCOUNTBLOB':'blob','HLTACCEPTBLOB':'blob'}
        for cmslsnum,perlsdata in toinsert.items():
            prescaleArray=array.array('I')
            hltcountArray=array.array('I')
            hltacceptArray=array.array('I')
            for (pathidx,hltname) in hltnamedict:
                thispathIdx=inputpathnames.index(hltname)
                thispresc=perlsdata[thispathIdx]
                thiscount=0
                thisaccept=0
                prescaleArray.append(thispresc)
                hltcountArray.append(thiscount)
                hltacceptArray.append(thisaccept)
            prescaleblob=CommonUtil.packArraytoBlob(prescaleArray)
            hltcountblob=CommonUtil.packArraytoBlob(hltcountArray)
            hltacceptblob=CommonUtil.packArraytoBlob(hltacceptArray)
            tabrowValueDict={'DATA_ID':oldhltdataid,'RUNNUM':int(runnum),'CMSLSNUM':int(cmslsnum),'PRESCALEBLOB':prescaleblob,'HLTCOUNTBLOB':hltcountblob,'HLTACCEPTBLOB':hltacceptblob}
            db=dbUtil.dbUtil(dbsession.nominalSchema())
            db.insertOneRow(nameDealer.lshltTableName(),tabrowDefDict,tabrowValueDict)
            #
            # update lshlt set prescaleblob=:prescaleblob,hltcoutblob=:hltcountblob,hltacceptblob=:hltacceptblob where data_id=:olddata_id and cmslsnum=:cmslsnum;
            #
        setClause='PRESCALEBLOB=:prescaleblob,HLTCOUNTBLOB=:hltcountblob,HLTACCEPTBLOB=:hltacceptblob'
        updateCondition='DATA_ID=:oldhltdataid and CMSLSNUM=:cmslsnum'
        for cmslsnum,perlsdata in toupdate.items():
            prescaleArray=array.array('I')
            hltcountArray=array.array('I')
            hltacceptArray=array.array('I')
            for (pathidx,hltname) in hltnamedict:
                thispathIdx=inputpathnames.index(hltname)
                thispresc=perlsdata[thispathIdx]
                thiscount=0
                thisaccept=0
                prescaleArray.append(thispresc)
                hltcountArray.append(thiscount)
                hltacceptArray.append(thisaccept)
            prescaleblob=CommonUtil.packArraytoBlob(prescaleArray)
            hltcountblob=CommonUtil.packArraytoBlob(hltcountArray)
            hltacceptblob=CommonUtil.packArraytoBlob(hltacceptArray)
            iData=coral.AttributeList()
            iData.extend('prescaleblob','blob')
            iData.extend('hltcountblob','blob')
            iData.extend('hltacceptblob','blob')
            iData.extend('olddata_id','unsigned int')
            iData.extend('cmslsnum','unsigned int')
            iData['prescaleblob'].setData(prescaleblob)
            iData['hltcountblob'].setData(hltcountblob)
            iData['hltacceptblob'].setData(hltacceptblob)
            iData['olddata_id'].setData(int(olddata_id))
            iData['cmslsnum'].setData(int(cmslsnum))
            db=dbUtil.dbUtil(schema)
            db.singleUpdate(nameDealer.lshltTableName(),setClause,updateCondition,iData)
        dbsession.transaction().commit()
        #dbsession.transaction().rollback()
    except :
        raise
    
def insertV2(dbsession,runnum,inputpathnames,inputdata):
    '''
    inputpathnames: [pathname]
    inputdata: {cmslsnum:[presc,presc...]}
    '''
    branchrevision_id=DATABRANCH_ID
    try:
        pathnamesClob=','.join(inputpathnames)
        hltrundata=[pathnamesClob,'text file']
        (hltrevid,hltentryid,hltdataid)=dataDML.addHLTRunDataToBranch(dbsession.nominalSchema(),runnum,hltrundata,(branchrevision_id,'DATA'))
        hltlsdata={}
        for cmslsnum,perlsdata in inputdata.items():
            prescaleArray=array.array('I')
            hltcountArray=array.array('I')
            hltacceptArray=array.array('I')
            for presc in perlsdata:
                thiscount=0
                thisaccept=0
                prescaleArray.append(presc)
                hltcountArray.append(thiscount)
                hltacceptArray.append(thisaccept)
            prescaleblob=CommonUtil.packArraytoBlob(prescaleArray)
            hltcountblob=CommonUtil.packArraytoBlob(hltcountArray)
            hltacceptblob=CommonUtil.packArraytoBlob(hltacceptArray)
            hltlsdata[cmslsnum]=[hltcountblob,hltacceptblob,prescaleblob]
        dataDML.bulkInsertHltLSData(dbsession,runnum,hltdataid,hltlsdata,500)
    except:
        raise
def parsepresc(inputlistoflist,minlsnum,maxlsnum,lsboundaries):
    '''
    input:
        inputlistoflist : [[pathidx,pathname,presc1,presc2],[]...]
        minlsnum: minimum ls
        maxlsnum: max ls
        prescboundaries=[(1,2),(3,67),(68,188),(189,636),(637,1004))]
    output:
        (hltpathnames,{cmslsnum:[presc]})
    '''
    if not maxlsnum:
        maxlsnum=lsboundaries[-1][1]
    else:
        maxlsnum=int(maxlsnum)
    if not minlsnum:
        minlsnum=lsboundaries[0][0]
    else:
        minlsnum=int(minlsnum)
    pathnames=[]
    dataresult={}#{cmslsnum:[presc...]}
    alllsnum=range(minlsnum,maxlsnum+1)
    prescidxdict={}#{cmsls:prescidx}
    for cmsls in alllsnum:
        dataresult[cmsls]=[]
        for prescidx,(blow,bhigh) in enumerate(lsboundaries):
            if cmsls>=blow and cmsls<=bhigh:
                prescidxdict[cmsls]=prescidx+2
    sortedinputlistoflist=sorted(inputlistoflist,key=lambda x: x[0])
    for pathinfo in sortedinputlistoflist:#loop over path
        pathindexfield=pathinfo[0]
        pathnamefield=pathinfo[1]
        pathname=pathnamefield.split(' ')[0]
        pathnames.append(pathname)
        for cmsls in alllsnum:
            for idx,p in enumerate(pathinfo):#loop over presc possibilities
                if idx==0 or idx==1: continue #this is a pathidx or pathname field
                presc=int(p)
                if idx==prescidxdict[cmsls]:
                    dataresult[cmsls].append(presc)
                    break
    return (pathnames,dataresult)
def parseInfile(filename):
    '''
    input filename
    output:(runnum,[(lsboundaryLow,lsboundaryHigh),...],[[pathidx,pathname,presc1,presc2,presc3...],[]])
             every path with its possible presc
    '''
    result=[]
    f=open(filename,'rb')
    reader=csv.reader(f,delimiter=',')
    i=0
    runnum=0
    lsboundaries=[]
    pathinfo=[]
    p=re.compile('\d+')
    datarownumber=0
    for row in reader:
        if not row or len(row)==0: continue
        fields=[r.strip() for r in row if r and len(r)!=0]
        if len(fields)==0: continue
        if i==0:
            runnum=row[0].split(' ')[1]
        elif i==1:
            for field in fields:
                [low,high]=p.findall(field)
                lsboundaries.append([int(low),int(high)])
        else:
            pathinfo.append([datarownumber]+fields)
            datarownumber+=1
        i+=1
    return (int(runnum),lsboundaries,pathinfo)

def  main(*args):
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Patch HLT prescale from text file",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['v2']
    parser.add_argument('action',choices=allowedActions,
                        help='command actions')
    parser.add_argument('-c',dest='connect',action='store',
                        required=True,
                        help='connect string to lumiDB,optional',
                        default=None)
    parser.add_argument('-P',dest='authpath',action='store',
                        required=True,
                        help='path to authentication file')
    parser.add_argument('-i',dest='ifile',action='store',
                        required=True,
                        help='patch data file ')
    parser.add_argument('-lsmin',dest='lsmin',action='store',
                        default=1,
                        required=False,
                        help='minimum ls to patch')
    parser.add_argument('-lsmax',dest='lsmax',action='store',
                        default=None,
                        required=False,
                        help='max ls to patch')
    parser.add_argument('--debug',dest='debug',action='store_true',
                        required=False,
                        help='debug ')
    options=parser.parse_args()
    (runnum,lsboundaries,pathinfo)=parseInfile(options.ifile)
    (pathnames,dataresult)=parsepresc(pathinfo,options.lsmin,options.lsmax,lsboundaries)
    print pathnames
    os.environ['CORAL_AUTH_PATH'] = options.authpath      
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    dbsession=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    patchV2(dbsession,runnum,pathnames,dataresult)
    del dbsession
    del svc
    
if __name__=='__main__':
    sys.exit(main(*sys.argv))

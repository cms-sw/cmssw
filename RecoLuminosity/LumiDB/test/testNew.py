#!/usr/bin/env python
VERSION='2.00'
import os,sys,array
import coral
from RecoLuminosity.LumiDB import sessionManager,dataDML,CommonUtil
DATABRANCH_ID=3
if __name__=='__main__':
    myconstr='oracle://cms_orcoff_prod/cms_lumi_prod'
    svc=sessionManager.sessionManager(myconstr,authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    session.transaction().start(True)
    schema=session.nominalSchema()
    runs=dataDML.runList(schema,fillnum=1616,runmin=160403,runmax=160957,nominalEnergy=3500.0)
    print runs
    myrun=runs[0]
    runsummary=dataDML.runsummary(schema,myrun)
    print runsummary
    normid=dataDML.guessnormIdByContext(schema,'PROTPHYS',3500)
    normval=dataDML.luminormById(schema,normid)[2]
    print 'norm in use ',normval
    (lumidataid,trgdataid,hltdataid)=dataDML.guessDataIdByRun(schema,myrun)
    print 'all dataids ',lumidataid,trgdataid,hltdataid
    (runnum,lumiLSdata)=dataDML.lumiLSById(schema,lumidataid)
    
    lumils=lumiLSdata.keys()
    lumils.sort()
    for lsnum in lumils:
        print 'lumilsnum,cmslsnum,instlumi ',lsnum,lumiLSdata[lsnum][0],lumiLSdata[lsnum][1]*normval
        
    (runnum,trgLSdata)=dataDML.trgLSById(schema,trgdataid)
    cmsls=trgLSdata.keys()
    cmsls.sort()
    for lsnum in cmsls:
        print 'cmslsnum,deadtime,bizerocount,bitzeroprescale,deadfrac ',lsnum,trgLSdata[lsnum][0],trgLSdata[lsnum][1],trgLSdata[lsnum][2],trgLSdata[lsnum][3]
    [runnum,datasource,npath,pathnames]=dataDML.hltRunById(schema,hltdataid)
    print 'npath,pathnames ',npath,pathnames
    pathnameList=pathnames.split(',')
    (runnum,hltLSdata)=dataDML.hltLSById(schema,hltdataid)
    cmsls=hltLSdata.keys()
    cmsls.sort()
    for lsnum in cmsls:
        prescaleblob=hltLSdata[lsnum][0]
        print 'lsnum ',lsnum
        if prescaleblob:
            hltprescales=CommonUtil.unpackBlobtoArray(prescaleblob,'h')
            for pidx,pathname in enumerate(pathnameList):
                print 'pathname, hltprescales ',pathname,hltprescales[pidx]
    session.transaction().commit()
    del session
    

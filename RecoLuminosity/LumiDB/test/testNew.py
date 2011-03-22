#!/usr/bin/env python
VERSION='2.00'
import os,sys,array
import coral
from RecoLuminosity.LumiDB import sessionManager,dataDML,revisionDML
DATABRANCH_ID=3
if __name__=='__main__':
    myconstr='oracle://cms_orcoff_prod/cms_lumi_prod'
    svc=sessionManager.sessionManager(myconstr,authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    session.transaction().start(True)
    schema=session.nominalSchema()
    runs=dataDML.runList(schema,fillnum=1616,runmin=160403,runmax=160957)
    session.transaction().commit()
    del session
    print runs

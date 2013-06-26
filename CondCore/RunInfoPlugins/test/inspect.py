import os,sys, DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
rdbms = RDBMS()

dbName =  "oracle://cms_orcoff_prod/CMS_COND_21X_RUN_INFO"
logName = "oracle://cms_orcoff_prod/CMS_COND_21X_POPCONLOG"

rdbms.setLogger(logName)
from CondCore.Utilities import iovInspector as inspect

db = rdbms.getDB(dbName)
tags = db.allTags()

tag = 'l1triggerscaler_test_v2'

try :
    log = db.lastLogEntry(tag)
    print log.getState()
    iov = inspect.Iov(db,tag)
    print iov.list()
    for x in  iov.summaries():
        print x[1],x[3]
    #        print iov.trend("",[0,2,12])
except RuntimeError :
    print " no iov? in", tag



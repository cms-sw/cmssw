import os,sys, DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
#os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb")

dbName =  "oracle://cms_orcoff_prod/CMS_COND_31X_RUN_INFO"
logName = "oracle://cms_orcoff_prod/CMS_COND_31X_POPCONLOG"

#rdbms.setLogger(logName)
from CondCore.Utilities import iovInspector as inspect

db = rdbms.getDB(dbName)
tags = db.allTags()
print  "########overview of all tags########"
print tags
# for inspecting last run after run has started  
tag = 'runinfo_31X_hlt'

# for inspecting last run after run has stopped  
#tag = 'runinfo_test'

try :
    #log = db.lastLogEntry(tag)

    #for printing all log info present into log db 
    #print log.getState()

    iov = inspect.Iov(db,tag)
    print "########overview of tag "+tag+"########"
    print iov.list()
    #for x in  iov.summaries():
    #    print x[1], x[2] ,x[3]
    what={}
    print "###(start_current,stop_current,avg_current,max_current,min_current,run_interval_micros) vs runnumber###"
    print iov.trend(what)
    print "########(start_current,stop_current,avg_current,max_current,min_current,run_interval_micros) vs runnumber in a given range########"
    print iov.trendinrange(what,109441L,109503L)
except Exception, er :
    print er



#
# I should write a decent test of the python binding...
#
def dumpSummaries(dbname):
    db = rdbms.getDB(dbName)
    tags = db.allTags()
    
    for tag in tags.split() :
        try :
            #        log = db.lastLogEntry(tag)
            #        print log.getState()
            iov = inspect.Iov(db,tag)
            #        print iov.list()
            for x in  iov.summaries():
                print x[1],x[2],x[3]
            #        print iov.trend("",[0,2,12])
        except RuntimeError :
            print " no iov? in", tag
    
    iov=0



import sys, DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
rdbms = RDBMS()

dbName = "sqlite_file:mydb.db"

# dbName =  "oracle://cms_orcoff_prod/CMS_COND_20X_ECAL"
#logName = "oracle://cms_orcoff_prod/CMS_COND_21X_POPCONLOG"

#rdbms.setLogger(logName)
from CondCore.Utilities import iovInspector as inspect

dumpSummaries(dbName)

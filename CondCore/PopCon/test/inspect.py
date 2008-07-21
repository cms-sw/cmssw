import sys, DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
rdbms = RDBMS()

logName = "sqlite_file:log.db"
dbName = "sqlite_file:pop_test.db"

dbName =  "oracle://cms_orcoff_prod/CMS_COND_20X_ECAL"
logName = "oracle://cms_orcoff_prod/CMS_COND_21X_POPCONLOG"

rdbms.setLogger(logName)
from CondCore.Utilities import iovInspector as inspect

db = rdbms.getDB(dbName)
tags = db.allTags()


for tag in tags.split() :
    try :
        log = db.lastLogEntry(tag)
        print log.getState()
        iov = inspect.Iov(db,tag)
        print iov.list()
#        print iov.summaries()
#        print iov.trend("",[0,2,12])
    except RuntimeError :
        print " no iov? in", tag


iov=0

tag = tags.split()[0]

token = log.payloadToken

p = inspect.PayLoad(db,token)
print p

p=0

p = db.payLoad(token)
o = Plug.Object(p)
o.summary()
o.dump()
o=0

tag = tags.split()[0]
iov = inspect.Iov(db,tag)
iov.summaries()
iov.trend("",[0,2,12])



o = iovInspector.PayLoad(db,log.payloadToken)

        exec('import '+db.moduleName(tag)+' as Plug')   
        iov = db.iov(tag)
        log = db.lastLogEntry(tag)
        print tag, iov.size(), log.execmessage, log.exectime, log.payloadIdx 
        vi = VInt()
        vi.append(0)
        vi.append(2)
        vi.append(12)
        ex = Plug.Extractor("",vi)
        for elem in iov.elements :
            p = Plug.Object(elem)
            print elem.since(), elem.till(),p.summary()
            p.extract(ex)
            for v in ex.values() :
                print v

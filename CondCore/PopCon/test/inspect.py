import DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
rdbms = RDBMS()
db = rdbms.getDB("sqlite_file:pop_test.db")
tags = db.allTags()
for tag in tags.split() :
    try :
        iov = db.iov(tag)
        print tag, iov.size()
        for elem in iov.elements :
            print elem.since(), elem.till(), elem.payloadToken()
    except RuntimeError :
        print " no iov?"
        
iov=0



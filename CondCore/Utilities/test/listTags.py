import DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
rdbms = RDBMS()
db = rdbms.getDB("oracle://cms_orcoff_prod/CMS_COND_20X_ALIGNMENT")
tags = db.allTags()
for tag in tags.split() :
    try :
        iov = db.iov(tag)
        print tag, iov.size()
        iov = 0
    except RuntimeError :
        print " no iov?"

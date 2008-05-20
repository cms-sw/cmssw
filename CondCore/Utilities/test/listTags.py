import DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
rdbms = RDBMS()
dba = rdbms.getDB("oracle://cms_orcoff_prod/CMS_COND_20X_ALIGNMENT")
dbe = rdbms.getDB("oracle://cms_orcoff_prod/CMS_COND_20X_ECAL")
for db in (dba,dbe) :
    tags = db.allTags()
    for tag in tags.split() :
        try :
            iov = db.iov(tag)
            print tag, iov.size()
            for elem in iov.elements :
                print elem.since(), elem.till(), elem.payloadToken()
        except RuntimeError :
            print " no iov?"


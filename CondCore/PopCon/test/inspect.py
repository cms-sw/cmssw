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
        exec('import '+db.moduleName(tag)+' as Plug')   
        iov = db.iov(tag)
        print tag, iov.size()
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
    except RuntimeError :
        print " no iov?"
        
iov=0



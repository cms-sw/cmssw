import DLFCN, sys, os, time
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
#os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")



# rdbms = RDBMS()
rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb")
logName = "oracle://cms_orcoff_prod/CMS_COND_31X_POPCONLOG"
gdbName = "oracle://cms_orcoff_prod/CMS_COND_31X_GLOBALTAG"
# gName = "GR09_31X_V6P::All"
gName = 'CRAFT09_R_V9::All'
#gName = 'STARTUP3X_V8F::All'
rdbms.setLogger(logName)
#globalTag = rdbms.globalTag(gdbName,gName,"","")
# globalTag = rdbms.globalTag(gdbName,gName,"oracle://cms_orcoff_prod/","")
globalTag = rdbms.globalTag(gdbName,gName,"frontier://FrontierArc/","_0911")

for tag in globalTag.elements:
#    dbname = tag.pfn[tag.pfn.rfind('/'):]
#    db_o = rdbms.getDB("oracle://cms_orcoff_prod"+dbname)
    db = rdbms.getDB(tag.pfn)
    log = db.lastLogEntry(tag.tag)
    iov = db.iov(tag.tag)
    iov.tail(1)
    for elem in iov.elements :
        lastSince = elem.since()
    print tag.record,tag.label,tag.pfn,tag.tag
    print iov.size(), iov.revision(), time.asctime(time.gmtime(unpackTime(iov.timestamp())[0])), iov.comment(), lastSince
    print log.getState()
    iov=0
    db=0


def iovSize(rdbms,conn,tag) :
    try :
        db = rdbms.getDB(conn)
        log = db.lastLogEntry(tag)
        iov = db.iov(tag)
        size = iov.size()
        for elem in iov.elements :
            if (elem.till()>4294967295) : 
                print tag, elem.since(), elem.till(), elem.payloadToken()
    except RuntimeError :
        print conn, tag," no iov?"
        size=-1
    iov=0
    db=0
    return size





import DLFCN, sys, os
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)
import pluginCondDBPyInterface as condDB
a = condDB.FWIncantation()
rdbms = condDB.RDBMS()
conn = "frontier://FrontierPrep/CMS_COND_STRIP" # for develoment DB
conn = "frontier://FrontierInt/CMS_COND_STRIP" # for integration DB (as agreed to do tests)
conn = "frontier://PromptProd/CMS_COND_31X_STRIP"
db = rdbms.getReadOnlyDB(conn)
tag = "SiStripLatency_GR10_v2_hlt"
db.startReadOnlyTransaction()
iov = db.iov(tag)
# print list(db.payloadModules(tag))
Plug = __import__(str(db.payloadModules(tag)[0]))
payload = Plug.Object(db)
listOfIovElem= [iovElem for iovElem in iov.elements]

runNumber = 0

if len(sys.argv) >= 2:
	runNumber = int(sys.argv[1])

for elem in iov.elements:
    if runNumber==0 or (runNumber >= elem.since() and runNumber <= elem.till()):
        theIOV = payload.load(elem)
        if theIOV:
            payload.dumpXML("dump_"+str(elem.since())+".xml")
            if payload.summary().find("PEAK") != -1:
                print "since =", elem.since(), ", till =", elem.till(), "--> peak mode"
            elif payload.summary().find("DECO") != -1:
                print "since =", elem.since(), ", till =", elem.till(), "--> deco mode"
            else:
                print "since =", elem.since(), ", till =", elem.till(), "--> mixed mode"
        else:
            print "error in retriving payload"
            
db.commitTransaction()

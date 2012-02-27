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

if len(sys.argv) < 2:
    print "Please specify the IOV (run number)"
    sys.exit()

runNumber = int(sys.argv[1])

for elem in iov.elements:
    # print elem.since()
    # print elem.till()
    if runNumber >= elem.since() and runNumber <= elem.till():
        theIOV = payload.load(elem)
        print "since =", elem.since(), ", till =", elem.till()
        if theIOV:
            payload.dumpXML("dump_"+str(elem.since())+".xml")
            # if payload.summary().find("All the Tracker has the same mode = 47") != -1:
            if payload.summary().find("the same mode = 47") != -1:
                print "peak mode"
            # elif payload.summary().find("All the Tracker has the same mode = 37") != -1:
            elif payload.summary().find("the same mode = 37") != -1:
                print "deco mode"
            else:
                print "mixed mode"
            break
        else:
            print "error in retriving payload"
db.commitTransaction()

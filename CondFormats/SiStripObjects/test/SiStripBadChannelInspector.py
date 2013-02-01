import DLFCN, sys, os
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)
import pluginCondDBPyInterface as condDB
a = condDB.FWIncantation()
rdbms = condDB.RDBMS()
conn = "frontier://FrontierPrep/CMS_COND_STRIP" # for develoment DB
conn = "frontier://FrontierInt/CMS_COND_STRIP" # for integration DB (as agreed to do tests)
conn = "frontier://PromptProd/CMS_COND_31X_STRIP"
db = rdbms.getReadOnlyDB(conn)
tag = "SiStripBadChannel_FromOfflineCalibration_GR10_v1_prompt"
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
            # print payload.summary()
            print payload.summary()
            print payload.dump()
db.commitTransaction()

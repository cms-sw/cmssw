#
# I should write a decent test of the python binding...
#
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

vi = VInt()
vf = VFloat()

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

tag='EcalPedestals_online'
what = inspect.extractorWhat(db,tag)
ans = {}
for key in what.keys():
    (mode,val) = what[key]
    if (mode=='single') :
        ans[key]=val[2]
    else :
        ans[key]=[0,2,12]


iov = inspect.Iov(db,tag)
print iov.trend(ans)
ans = {'how': 'all', 'quantity': 'mean_x3', 'which' : []}
print iov.trend(an)
ans = {'how': 'singleChannel', 'quantity': 'mean_x6', 'which' : [0,200,1200]}
print iov.trend(ans)




# tag = tags.split()[0]

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


token = '[DB=00000000-0000-0000-0000-000000000000][CNT=EcalPedestalsRcd][CLID=75E7B995-8233-097B-FD4A-31AEC6A040C8][TECH=00000B01][OID=0000000C-00000114]'
p = inspect.PayLoad(db,token)

import DLFCN, sys, os
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")

import coral
from CondCore.TagCollection import Node,tagInventory,TagTree
#context = coral.Context()
#context.setVerbosityLevel( 'ERROR' )
# context.setVerbosityLevel( 'DEBUG' )
svc = coral.ConnectionService()
session = svc.connect("sqlite_file:CondCore/TagCollection/data/GlobalTag.db",accessMode = coral.access_ReadOnly )
# session = svc.connect("oracle://cms_orcoff_prod/CMS_COND_31X_GLOBALTAG",accessMode = coral.access_ReadOnly )
inv=tagInventory.tagInventory(session)
# mytree=TagTree.tagTree(session,"GR09_31X_V5P")
mytree=TagTree.tagTree(session,"GR09_31X")
result=mytree.getAllLeaves()
tags=[]
for r in result:
    if r.tagid != 0:
        tagcontent=inv.getEntryById(r.tagid)
        tags.append((tagcontent.recordname, tagcontent.labelname, tagcontent.tagname,tagcontent.pfn))


mytree=0
inv=0
del session
del svc
rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb")


# cmscond_list_iov -c "frontier://(proxyurl=http://cmst0frontier1.cern.ch:3128)(serverurl=http://cmsfrontier.cern.ch:8000/FrontierProd)(forcereload=short)/CMS_COND_20X_HCAL"



def iovSize(rdbms,conn,tag) :
    try :
        db = rdbms.getDB(conn)
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


frontiers=[
    "frontier://PromptProd",
    "frontier://FrontierProd",
    "frontier://(serverurl=http://cmsfrontier.cern.ch:8000/FrontierProd)",
    "frontier://(serverurl=http://cmsfrontier1.cern.ch:8000/FrontierProd)",
    "frontier://(serverurl=http://cmsfrontier2.cern.ch:8000/FrontierProd)",
    "frontier://(serverurl=http://cmsfrontier3.cern.ch:8000/FrontierProd)",
    "frontier://(serverurl=http://cmsfrontier4.cern.ch:8000/FrontierProd)",
    "frontier://(proxyurl=http://cmst0frontier1.cern.ch:3128)(serverurl=http://cmsfrontier.cern.ch:8000/FrontierProd)",
    "frontier://(proxyurl=http://cmst0frontier2.cern.ch:3128)(serverurl=http://cmsfrontier.cern.ch:8000/FrontierProd)",
    "frontier://(proxyurl=http://cmst0frontier3.cern.ch:3128)(serverurl=http://cmsfrontier.cern.ch:8000/FrontierProd)",
    "frontier://(proxyurl=http://cmst0frontier.cern.ch:3128)(serverurl=http://cmsfrontier.cern.ch:8000/FrontierProd)"
    ]

for tag in tags:
    dbname = tag[1][tag[1].rfind('/'):]
    # db_o = rdbms.getDB(tag[1])
    # db_o = rdbms.getDB("oracle://cms_orcoff_prod"+dbname)
    size_o = iovSize(rdbms,"oracle://cms_orcoff_prod"+dbname,tag[0])
    size_f = []
    for f in frontiers:
        size =  iovSize(rdbms,f+dbname,tag[0])
        size_f.append(size)
        if (size!=size_o):
            print tag[0], 'not updated in',  f, size, size_o
            if (f.find(')')!=-1) :
                size =  iovSize(rdbms,f+"(forcereload=short)"+dbname,tag[0])
                if (size!=size_o):
                    print "update failed", size
    print tag[1], tag[0], size_o, size_f


# SiStripFedCabling_TKCC_20X_v3_hlt

rdbms = RDBMS()
dba = rdbms.getDB("oracle://cms_orcoff_prod/CMS_COND_20X_ALIGNMENT")
dbb = rdbms.getDB("frontier://FrontierProd/CMS_COND_20X_ALIGNMENT")
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



----------

conn = 'frontier://FrontierProd/CMS_COND_31X_RUN_INFO'
db = rdbms.getDB(conn)

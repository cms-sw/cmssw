import DLFCN, sys, os
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")

import coral
from CondCore.TagCollection import Node,tagInventory,TagTree
context = coral.Context()
# context.setVerbosityLevel( 'ERROR' )
context.setVerbosityLevel( 'DEBUG' )
svc = coral.ConnectionService( context )
session = svc.connect("oracle://cms_orcoff_prod/CMS_COND_20X_GLOBALTAG",accessMode = coral.access_ReadOnly )
inv=tagInventory.tagInventory(session)
mytree=TagTree.tagTree(session,"CRUZET3_V2P")
result=mytree.getAllLeaves()
tags=[]
for r in result:
    if r.tagid != 0:
        tagcontent=inv.getEntryById(r.tagid)
        tags.append((tagcontent.tagname,tagcontent.pfn))


mytree=0
inv=0
del session
del svc
rdbms = RDBMS()


frontiers=[
    "frontier://PromptProd",
    "frontier://cmsfrontier.cern.ch:8000/Frontier",
    "frontier://cmsfrontier1.cern.ch:8000/Frontier",
    "frontier://cmsfrontier2.cern.ch:8000/Frontier",
    "frontier://cmsfrontier3.cern.ch:8000/Frontier"
    ]

for tag in tags:
    db_o = rdbms.getDB(tag[1].replace('frontier://FrontierProd',"oracle:///cms_orcoff_prod"))
    try :
        iov_o = db_o.iov(tag[0])
        size_o = iov_o.size()
        iov_o=0
        size_f = []
        for f in frontiers:
            db_f = rdbms.getDB(tag[1].replace('frontier://FrontierProd',f))
            iov_f = db_f.iov(tag[0])
            size_f.append(iov_f.size())
            iov_f=0
        print tag[0], size_o, size_f
        #for elem in iov_o.elements :
        #    print elem.since(), elem.till(), elem.payloadToken()
    except RuntimeError :
        print tag," no iov?"



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


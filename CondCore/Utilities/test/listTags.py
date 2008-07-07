import DLFCN, sys, os
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")

import coral
from CondCore.TagCollection import Node,tagInventory,TagTree
context = coral.Context()
context.setVerbosityLevel( 'ERROR' )
svc = coral.ConnectionService( context )
session = svc.connect("oracle://cms_orcoff_prod/CMS_COND_20X_GLOBALTAG",accessMode = coral.access_ReadOnly )
inv=tagInventory.tagInventory(session)
mytree=TagTree.tagTree(session,"CRUZET_V3")
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
for tag in tags:
    db = rdbms.getDB(tag[1].replace('frontier://FrontierProd',"oracle://cms_orcoff_prod"))
    try :
        iov = db.iov(tag[0])
        print tag[0], iov.size()
        for elem in iov.elements :
            print elem.since(), elem.till(), elem.payloadToken()
        iov=0
    except RuntimeError :
        print tag," no iov?"



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


#!/bin/env python



from CMGTools.Production.cmgdbApi import CmgdbApi
import CMGTools.Production.eostools as eos
import pprint

db = CmgdbApi()
db.connect()

cols, rows = db.sql("select path_name, file_owner, number_files_bad, number_files_good from dataset_details where path_name like '%V5_4_0'")
# pprint.pprint(rows)

dead = []
good = []

nscanned = 0
for row in rows:
    # print row
    path = row[0]
    owner = row[1]
    nFiles = None
    if row[2] is not None and row[3] is not None:
        nFiles = row[2]+row[3]
    print path, nFiles
    dir = None
    if owner == 'cmgtools_group':
        dir = '/eos/cms/store/cmst3/group/cmgtools/CMG'+path
    elif owner == 'cmgtools':
        dir = '/eos/cms/store/cmst3/user/cmgtools/CMG'+path  
    dirpresent = False
    try:
        dirpresent = eos.isEOSFile( dir )
    except AttributeError:
        continue
    if not dirpresent:
        print 'Directory disappeared'
        dead.append(path)
    else:
        good.append(path)
    nscanned += 1
    # if nscanned == 10:
    #    break
    
for path in dead:
    print 'DEAD', path
for path in good:
    print 'GOOD', path

print 'num rows             = ', len(rows)
print 'num datasets scanned = ', nscanned
print 'num datasets dead    = ', len(dead)
print 'num datasets ok(?)   = ', len(good) 


    

eval `scramv1 runtime -sh`
echo Run from DDD:
date
cmsRun testDTGeometry_cfg.py > tcgDD.out 2>&1
date
echo Load the DB:
date
rm -f test.db
cmsRun testLoadDTDb.py > tcgLoad.out 2>&1
date
echo Run from DB:
date
cmsRun testDTGeometryFromDB_cfg.py > tcgDB.out 2>&1
date
echo Diff-ing the output:
diff tcgDD.out tcgDB.out
echo Done. 

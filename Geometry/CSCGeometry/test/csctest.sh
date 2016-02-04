eval `scramv1 runtime -csh`
mkdir csctestOUTPUT
mkdir csctestOUTPUT/fromDDD
mkdir csctestOUTPUT/fromDB
cd csctestOUTPUT/fromDDD
echo Run from DDD:
date
cmsRun ../../testCSCGeometry_cfg.py |& tee tcgDD.out
date
cd ../fromDB
echo Load the DB:
date
cmsRun ../../testLoadCSCDb.py
date
echo Run from DB:
date
cmsRun ../../testCSCGeometryFromDB_cfg.py |& tee tcgDB.out
date
echo Diff-ing the output:
diff ../fromDDD/tcgDD.out tcgDB.out
echo Done.

#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${LOCAL_TEST_DIR}/python/dump.py
F2=${LOCAL_TEST_DIR}/python/dumpDDShapes.py
F5=${LOCAL_TEST_DIR}/python/testDDAngularAlgorithm.py
F6=${LOCAL_TEST_DIR}/python/testDDDetectorESProducer.py
F7=${LOCAL_TEST_DIR}/python/testDDPseudoTrapShapes.py
F8=${LOCAL_TEST_DIR}/python/testDDSpecPars.py
F9=${LOCAL_TEST_DIR}/python/testDDVectors.py
F12=${LOCAL_TEST_DIR}/python/testShapes.py
F13=${LOCAL_TEST_DIR}/python/testNavigateGeometry.py
F14=${LOCAL_TEST_DIR}/python/testTGeoIterator.py
F15=${LOCAL_TEST_DIR}/python/testDDSpecParsFilterG4ProdCuts.py
F16=${LOCAL_TEST_DIR}/python/testDDSpecParsFilter.py
F19=${LOCAL_TEST_DIR}/python/testDDCompactView.py
F21=${LOCAL_TEST_DIR}/python/testGeometry2021.py
F22=${LOCAL_TEST_DIR}/python/testGeometry2021FromDB.py
F23=${LOCAL_TEST_DIR}/python/testDDFilteredView.py

echo " testing DetectorDescription/DDCMS"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"python UnitsCheck.py cms.xml\" ===="
python ${LOCAL_TEST_DIR}/python/UnitsCheck.py ${LOCAL_TEST_DIR}/data/testUnits.xml
echo "===== Test \"cmsRun dump.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun dumpDDShapes.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
echo "===== Test \"cmsRun testDDAngularAlgorithm.py\" ===="
(cmsRun $F5) || die "Failure using cmsRun $F5" $?
echo "===== Test \"cmsRun testDDDetectorESProducer.py\" ===="
(cmsRun $F6) || die "Failure using cmsRun $F6" $?
echo "===== Test \"cmsRun testDDPseudoTrapShapes.py\" ===="
(cmsRun $F7) || die "Failure using cmsRun $F7" $?
echo "===== Test \"cmsRun testDDSpecPars.py\" ===="
(cmsRun $F8) || die "Failure using cmsRun $F8" $?
echo "===== Test \"cmsRun testDDVectors.py\" ===="
(cmsRun $F9) || die "Failure using cmsRun $F9" $?
echo "===== Test \"cmsRun testShapes.py\" ===="
(cmsRun $F12) || die "Failure using cmsRun $F12" $?
echo "===== Test \"cmsRun testNavigateGeometry.py\" ===="
(cmsRun $F13) || die "Failure using cmsRun $F13" $?
echo "===== Test \"cmsRun testTGeoIterator.py\" ===="
(cmsRun $F14) || die "Failure using cmsRun $F14" $?
echo "===== Test \"cmsRun testDDSpecParsFilterG4ProdCuts.py\" ===="
(cmsRun $F15) || die "Failure using cmsRun $F15" $?
echo "===== Test \"cmsRun testDDSpecParsFilter.py\" ===="
(cmsRun $F16) || die "Failure using cmsRun $F16" $?
echo "===== Test \"cmsRun testDDCompactView.py\" ===="
(cmsRun $F19) || die "Failure using cmsRun $F19" $?
echo "===== Test \"cmsRun testGeometry2021.py\" ===="
(cmsRun $F21) || die "Failure using cmsRun $F21" $?
echo "===== Test \"cmsRun testGeometry2021FromDB.py\" ===="
(cmsRun $F22) || die "Failure using cmsRun $F22" $?
echo "===== Test \"cmsRun testDDFilteredView.py\" ===="
(cmsRun $F23) || die "Failure using cmsRun $F23" $?

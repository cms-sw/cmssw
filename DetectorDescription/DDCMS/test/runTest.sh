#!/bin/sh
echo " testing DetectorDescription/DDCMS"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
python ${LOCAL_TEST_DIR}/python/UnitsCheck.py ${LOCAL_TEST_DIR}/data/cms.xml
cmsRun ${LOCAL_TEST_DIR}/python/dump.py
cmsRun ${LOCAL_TEST_DIR}/python/dumpDDShapes.py
cmsRun ${LOCAL_TEST_DIR}/python/dumpMFGeometry.py
cmsRun ${LOCAL_TEST_DIR}/python/testDDAngularAlgorithm.py
cmsRun ${LOCAL_TEST_DIR}/python/testDDAngularAlgorithm.py
cmsRun ${LOCAL_TEST_DIR}/python/testDDDetectorESProducer.py
cmsRun ${LOCAL_TEST_DIR}/python/testDDPseudoTrapShapes.py
cmsRun ${LOCAL_TEST_DIR}/python/testDDSpecPars.py
cmsRun ${LOCAL_TEST_DIR}/python/testDDVectors.py
cmsRun ${LOCAL_TEST_DIR}/python/testMFGeometry.py
cmsRun ${LOCAL_TEST_DIR}/python/testMuonGeometry.py
cmsRun ${LOCAL_TEST_DIR}/python/testShapes.py

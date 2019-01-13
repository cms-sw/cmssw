#!/bin/sh
echo " testing DetectorDescription/DDCMS"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"python UnitsCheck.py cms.xml\" ===="
python ${LOCAL_TEST_DIR}/python/UnitsCheck.py ${LOCAL_TEST_DIR}/data/cms.xml
echo "===== Test \"cmsRun dump.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/dump.py
echo "===== Test \"cmsRun dumpDDShapes.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/dumpDDShapes.py
echo "===== Test \"cmsRun dumpMFGeometry.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/dumpMFGeometry.py
echo "===== Test \"cmsRun testDDAngularAlgorithm.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testDDAngularAlgorithm.py
echo "===== Test \"cmsRun testDDAngularAlgorithm.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testDDAngularAlgorithm.py
echo "===== Test \"cmsRun testDDDetectorESProducer.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testDDDetectorESProducer.py
echo "===== Test \"cmsRun testDDPseudoTrapShapes.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testDDPseudoTrapShapes.py
echo "===== Test \"cmsRun testDDSpecPars.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testDDSpecPars.py
echo "===== Test \"cmsRun testDDVectors.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testDDVectors.py
echo "===== Test \"cmsRun testMFGeometry.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testMFGeometry.py
echo "===== Test \"cmsRun testMuonGeometry.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testMuonGeometry.py
echo "===== Test \"cmsRun testShapes.py\" ===="
cmsRun ${LOCAL_TEST_DIR}/python/testShapes.py

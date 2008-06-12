#!/bin/sh
rm -f ${LOCAL_TEST_DIR}/prod1.root ${LOCAL_TEST_DIR}/prod2.root ${LOCAL_TEST_DIR}/prodmerge.root
rm -f prod1.root prod2.root prodmerge.root
cmsRun ${LOCAL_TEST_DIR}/ProdTest1.cfg
cmsRun ${LOCAL_TEST_DIR}/ProdTest2.cfg
cmsRun ${LOCAL_TEST_DIR}/ProdTestMerge.cfg

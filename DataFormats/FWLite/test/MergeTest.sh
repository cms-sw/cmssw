#!/bin/sh
rm -f ${LOCAL_TEST_DIR}/prod1.root ${LOCAL_TEST_DIR}/prod2.root ${LOCAL_TEST_DIR}/prodmerge.root
rm -f prod1.root prod2.root prodmerge.root
cmsRun ${LOCAL_TEST_DIR}/ProdTest1_cfg.py
cmsRun ${LOCAL_TEST_DIR}/ProdTest2_cfg.py
cmsRun ${LOCAL_TEST_DIR}/ProdTestMerge_cfg.py

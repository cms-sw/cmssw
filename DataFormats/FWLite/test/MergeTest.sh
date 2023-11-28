#!/bin/sh
rm -f ${SCRAM_TEST_PATH}/prod1.root ${SCRAM_TEST_PATH}/prod2.root ${SCRAM_TEST_PATH}/prodmerge.root
rm -f prod1.root prod2.root prodmerge.root
cmsRun ${SCRAM_TEST_PATH}/ProdTest1_cfg.py
cmsRun ${SCRAM_TEST_PATH}/ProdTest2_cfg.py
cmsRun ${SCRAM_TEST_PATH}/ProdTestMerge_cfg.py

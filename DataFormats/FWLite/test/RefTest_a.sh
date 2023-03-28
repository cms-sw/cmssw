#!/bin/sh
rm -f ${SCRAM_TEST_PATH}/good_a.root ${SCRAM_TEST_PATH}/good_b.root ${SCRAM_TEST_PATH}/empty_a.root
rm -f good_a.root good_b.root empty_a.root
cmsRun ${SCRAM_TEST_PATH}/RefTest_a_cfg.py
cmsRun ${SCRAM_TEST_PATH}/RefTest_b_cfg.py
cmsRun ${SCRAM_TEST_PATH}/EmptyFile_a_cfg.py

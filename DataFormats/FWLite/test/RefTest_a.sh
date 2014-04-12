#!/bin/sh
rm -f ${LOCAL_TEST_DIR}/good_a.root ${LOCAL_TEST_DIR}/good_b.root ${LOCAL_TEST_DIR}/empty_a.root
rm -f good_a.root good_b.root empty_a.root
cmsRun ${LOCAL_TEST_DIR}/RefTest_a_cfg.py
cmsRun ${LOCAL_TEST_DIR}/RefTest_b_cfg.py
cmsRun ${LOCAL_TEST_DIR}/EmptyFile_a_cfg.py

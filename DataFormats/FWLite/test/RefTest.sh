#!/bin/sh
rm -f ${LOCAL_TEST_DIR}/good.root
rm -f ${LOCAL_TEST_DIR}/good2.root
rm -f good.root
rm -f good2.root
cmsRun ${LOCAL_TEST_DIR}/RefTest_cfg.py
cmsRun ${LOCAL_TEST_DIR}/RefTest2_cfg.py

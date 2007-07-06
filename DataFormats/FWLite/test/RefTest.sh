#!/bin/sh
rm -f ${LOCAL_TEST_DIR}/good.root
rm -f ${LOCAL_TEST_DIR}/good2.root
rm -f good.root
rm -f good2.root
cmsRun ${LOCAL_TEST_DIR}/RefTest.cfg
cmsRun ${LOCAL_TEST_DIR}/RefTest2.cfg

#!/bin/sh
rm -f ${SCRAM_TEST_PATH}/vectorinttest.root
rm -f vectorinttest.root
cmsRun ${SCRAM_TEST_PATH}/vip_cfg.py

#!/bin/sh
rm -f ${LOCAL_TEST_DIR}/vectorinttest.root
rm -f vectorinttest.root
cmsRun ${LOCAL_TEST_DIR}/vip_cfg.py

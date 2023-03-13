#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest1_cfg.py || die "cmsRun ThinningTest1_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest2_cfg.py || die "cmsRun ThinningTest2_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest3_cfg.py || die "cmsRun ThinningTest3_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest4_cfg.py || die "cmsRun ThinningTest4_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest_dropOnInput_cfg.py || die "cmsRun ThinningTest_dropOnInput_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTestSubProcess_cfg.py || die "cmsRun ThinningTestSubProcess_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTestSubProcessRead_cfg.py || die "cmsRun ThinningTestSubProcessRead_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTestStreamerIn_cfg.py || die "cmsRun ThinningTestStreamerIn_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest4Slimming_cfg.py || die "cmsRun ThinningTest4Slimming_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/DetSetVectorThinningTest1_cfg.py || die "cmsRun DetSetVectorThinningTest1_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/DetSetVectorThinningTest2_cfg.py || die "cmsRun DetSetVectorThinningTest2_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTestSiblings_cfg.py && die "cmsRun SlimmingTestSiblings_cfg.py"1

  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTestFartherSiblings_cfg.py && die "cmsRun SlimmingTestFartherSiblings_cfg.py" 1

  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest1_cfg.py || die "cmsRun SlimmingTest1_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2A_cfg.py || die "cmsRun SlimmingTest2A_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2B_cfg.py || die "cmsRun SlimmingTest2B_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2C_cfg.py || die "cmsRun SlimmingTest2C_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2D_cfg.py || die "cmsRun SlimmingTest2D_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2E_cfg.py || die "cmsRun SlimmingTest2E_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2F_cfg.py || die "cmsRun SlimmingTest2E_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2G_cfg.py || die "cmsRun SlimmingTest2G_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2H_cfg.py || die "cmsRun SlimmingTest2H_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest2I_cfg.py || die "cmsRun SlimmingTest2I_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest3B_cfg.py || die "cmsRun SlimmingTest3B_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest3C_cfg.py || die "cmsRun SlimmingTest3C_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest3D_cfg.py || die "cmsRun SlimmingTest3D_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest3E_cfg.py || die "cmsRun SlimmingTest3E_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest3F_cfg.py || die "cmsRun SlimmingTest3F_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest3I_cfg.py || die "cmsRun SlimmingTest3I_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest4B_cfg.py || die "cmsRun SlimmingTest4B_cfg.py" $?
  cmsRun -p ${LOCAL_TEST_DIR}/SlimmingTest4F_cfg.py && die "cmsRun SlimmingTest4F_cfg.py" 1

exit 0

#!/bin/sh -ex
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/PreRNTupleInputTest_cfg.py RunPerLumiTest.rntpl 50 1 25 1 5 || die 'Failure using PreRNTupleInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/RunPerLumiTest_cfg.py 25 >& RunPerLumiTest.txt || die 'Failure using RunPerLumiTest_cfg.py' $?
grep 'record' RunPerLumiTest.txt | cut -d ' ' -f 4-11 > RunPerLumiTest.filtered.txt
diff ${LOCAL_TEST_DIR}/unit_test_outputs/RunPerLumiTest.filtered.txt RunPerLumiTest.filtered.txt || die 'incorrect output using RunPerLumiTest_cfg.py' $? 

cmsRun ${LOCAL_TEST_DIR}/RunPerLumiTest_cfg.py 50 >& tooManyLumis.txt && die 'RunPerLumiTest_cfg.py should have failed but did not' 1
grep "MismatchedInputFiles" tooManyLumis.txt || die  'RunPerLumiTest_cfg.py should have failed but did not' $?

cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRunTest_cfg.py 'file:RunPerLumiTest.rntpl' 25 1 25 1 5 || die 'Failure using firstLuminosityBlockForEachRunTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/PreRNTupleInputTest_cfg.py firstLumiTest1.rntpl 25 1 100 1 5 || die 'Failure using PreRNTupleInputTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/PreRNTupleInputTest_cfg.py firstLumiTest2.rntpl 25 1 100 6 5 || die 'Failure using PreRNTupleInputTest_cfg.py' $?
#multiple files not yet supported in RNTupleSource
#cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRunTest_cfg.py 'file:firstLumiTest1.rntpl,file:firstLumiTest2.rntpl' 50 1 25 1 5 || die 'Failure using firstLuminosityBlockForEachRunTest_cfg.py with 2 files' $?
#cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRunTest_cfg.py 'file:firstLumiTest1.rntpl,file:firstLumiTest2.rntpl' 50 1 25 1 5 shareRun || die 'Failure using firstLuminosityBlockForEachRunTest_cfg.py with 2 files which share a run' $?

#skip lumis not supported yet in RNTupleSource
#cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py 'file:firstLumiTest1.rntpl' 2 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py with first lumi 2' $?
#cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py 'file:firstLumiTest1.rntpl' 4 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py with first lumi 4' $?
#cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py 'file:firstLumiTest1.rntpl' 3 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py with first lumi 3' $?

#cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py 'file:firstLumiTest1.rntpl' 2 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py with first lumi 2' $?
#cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py 'file:firstLumiTest1.rntpl' 4 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py with first lumi 4' $?
#cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py 'file:firstLumiTest1.rntpl' 3 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py with first lumi 3' $?

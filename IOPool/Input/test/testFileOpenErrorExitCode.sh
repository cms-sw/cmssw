#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

mkdir -p SITECONF
mkdir -p SITECONF/local
mkdir -p SITECONF/local/JobConfig

export SITECONFIG_PATH=${PWD}/SITECONF/local
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cp ${LOCAL_TEST_DIR}/sitelocalconfig/noFallbackFile/site-local-config.xml  ${SITECONFIG_PATH}/JobConfig/
cp ${LOCAL_TEST_DIR}/sitelocalconfig/noFallbackFile/local/storage.json ${SITECONFIG_PATH}/
F1=${LOCAL_TEST_DIR}/test_fileOpenErrorExitCode_cfg.py
cmsRun -j NoFallbackFile_jobreport.xml $F1 -- --input FileThatDoesNotExist.root && die "$F1 should have failed but didn't, exit code was 0" 1

CMSRUN_EXIT_CODE=$(edmFjrDump --exitCode NoFallbackFile_jobreport.xml)
echo "Exit code after first run of test_fileOpenErrorExitCode_cfg.py is ${CMSRUN_EXIT_CODE}"
if [ "x${CMSRUN_EXIT_CODE}" != "x8020" ]; then
  echo "Unexpected cmsRun exit code after FileOpenError, exit code from jobReport ${CMSRUN_EXIT_CODE} which is different from the expected 8020"
  exit 1
fi

cp ${LOCAL_TEST_DIR}/sitelocalconfig/useFallbackFile/site-local-config.xml  ${SITECONFIG_PATH}/JobConfig/
cp ${LOCAL_TEST_DIR}/sitelocalconfig/useFallbackFile/local/storage.json ${SITECONFIG_PATH}/
cmsRun -j UseFallbackFile_jobreport.xml $F1 -- --input FileThatDoesNotExist.root > UseFallbackFile_output.log 2>&1 && die "$F1 should have failed after file fallback but didn't, exit code was 0" 1
grep -q "Input file abc/store/FileThatDoesNotExist.root could not be opened, and fallback was attempted" UseFallbackFile_output.log
RET=$?
if [ "${RET}" != "0" ]; then
    cat UseFallbackFile_output.log
    die "UseFallbackFile_output.log did not contain the detailed error message of the first file open failure " $RET
fi

CMSRUN_EXIT_CODE=$(edmFjrDump --exitCode UseFallbackFile_jobreport.xml)
echo "Exit code after second run of test_fileOpenErrorExitCode_cfg.py is ${CMSRUN_EXIT_CODE}"
if [ "x${CMSRUN_EXIT_CODE}" != "x8028" ]; then
  echo "Unexpected cmsRun exit code after FallbackFileOpenError, exit code from jobReport ${CMSRUN_EXIT_CODE} which is different from the expected 8028"
  exit 1
fi

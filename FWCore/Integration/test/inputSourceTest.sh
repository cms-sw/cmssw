#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/inputSourceTest_cfg.py || die 'Failed in inputSourceTest_cfg.py' $?

cmsRun ${SCRAM_TEST_PATH}/testLateLumiClosure_cfg.py || die 'Failed in testLateLumiClosure_cfg.py' $?

# The following demonstrates declaring the last run or lumi entry to be merged eliminates the delay
# before globalBeginRun and globalBeginLumi while waiting for the next thing to arrive
# to know that the last entry to be merged has already arrived. (Note the previous test is very similar
# and shows the delay, running without enableDeclareLast will also demonstrate it).

cmsRun ${SCRAM_TEST_PATH}/testDeclareLastEntryForMerge_cfg.py --enableDeclareLast --multipleEntriesForRun 2 --multipleEntriesForLumi 4 || die 'Failed in testDeclareLastEntryForMerge_cfg.py' $?

# The next two cmsRun processes should throw an exception (intentional)
# These two tests show the Framework will detect a buggy InputSource that
# declares something last that is NOT last.

cmsRun ${SCRAM_TEST_PATH}/testDeclareLastEntryForMerge_cfg.py --enableDeclareAllLast --multipleEntriesForRun 1 && die 'Failed in testDeclareLastEntryForMerge_cfg.py, last run source bug not detected' 1

cmsRun ${SCRAM_TEST_PATH}/testDeclareLastEntryForMerge_cfg.py --enableDeclareAllLast --multipleEntriesForLumi 2 && die 'Failed in testDeclareLastEntryForMerge_cfg.py, last lumi source bug not detected' 1

exit 0

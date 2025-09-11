#!/bin/bash

test=inconsistent_products_

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
echo ${test}prod_allThings ------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --startEvent=11 --fileName='allThings.root' || die "cmsRun ${test}prod_cfg.py" $?

echo ${test}prod_dropThing2 ------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --dropThing2 --fileName 'dropThing2.root'|| die "cmsRun ${test}prod_cfg.py --dropThing2" $?

echo ${test}test_dropThing2------------------------------------------------------------
#If file which contains the branch is read second, things fail
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --fileNames dropThing2.root allThings.root && die "cmsRun ${test}cfg.py dropThing2 --fileNames dropThing2.root allThings.root" 1

echo ${test}test_dropThing2_2nd------------------------------------------------------------
#If file which contains the branch is read first, things succeed
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --thing2Dropped --fileNames allThings.root dropThing2.root  || die "cmsRun ${test}cfg.py --thing2Dropped --fileNames allThings.root dropThing2.root" $?

echo ${test}prod_dropThing2_other2 ------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --dropThing2 --addAndStoreOther2 --fileName 'dropThing2_other2.root'|| die "cmsRun ${test}prod_cfg.py --dropThing2 --addAndStoreOther2" $?

echo ${test}prod_allThings ------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --startEvent=11 --addAndStoreOther2 --fileName='allThings_other2.root' || die "cmsRun ${test}prod_cfg.py --addAndStoreOther2" $?

echo ${test}test_dropThing2_other2_allThings------------------------------------------------------------
#If file which contains the branch is read second, things fail
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --fileNames dropThing2_other2.root allThings.root && die "cmsRun ${test}cfg.py --fileNames dropThing2_other2.root allThings.root" 1

echo ${test}test_dropThing2_other2------------------------------------------------------------
#If file which contains the branch is read second, things fail
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --fileNames dropThing2_other2.root allThings_other2.root && die "cmsRun ${test}cfg.py --fileNames dropThing2_other2.root allThings_other2.root" 1

echo ${test}test_dropThing2_other2_second_allThingsFirst------------------------------------------------------------
#If file which contains the branch is read second, things fail, here the branch is other2
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --thing2Dropped --fileNames allThings.root dropThing2_other2.root && die "cmsRun ${test}cfg.py --thing2Dropped --fileNames allThings.root dropThing2_other2.root" 1

echo ${test}test_dropThing2_other2_second------------------------------------------------------------
#If file which contains the branch is read first, things succeed
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --thing2Dropped --other2Run --fileNames allThings_other2.root dropThing2_other2.root || die "cmsRun ${test}cfg.py --thing2Dropped --fileNames allThings_other2.root dropThing2_other2.root" $?

echo ${test}prod_noThing2_other2 ------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --noThing2Prod --addAndStoreOther2 --fileName 'noThing2_other2.root'|| die "cmsRun ${test}prod_cfg.py --dropThing2 --addAndStoreOther2" $?

echo ${test}test_noThing2_other2_second------------------------------------------------------------
#If file which contains the branch is read first, things succeed
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --thing2NotRun --other2Run --fileNames allThings_other2.root noThing2_other2.root || die "cmsRun ${test}cfg.py --thing2Dropped --fileNames allThings_other2.root noThing2_other2.root" $?

echo ${test}test_noThing2_other2_first------------------------------------------------------------
#If file which contains the branch is read second, things fails
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10 --other2Run --fileNames noThing2_other2.root allThings_other2.root && die "cmsRun ${test}cfg.py --thing2Dropped --fileNames noThing2_other2 allThings_other2.root" 1

echo ${test}prod_dropThing2_other2_start11 ------------------------------------------------------------
#if thing2 is dropped and we still run other2, then other2 will read from thing1 instead thereby changing the provenance
cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --startEvent=11 --dropThing2 --addAndStoreOther2  --fileName='dropThing2_other2_start11.root' || die "cmsRun ${test}prod_cfg.py --startEvent=11 --dropThing2 --addAndStoreOther2" $?

#branch structures are the same but in one other2 depends on thing1 while in the other file other2 depends on thing2
echo ${test}test_dropThing2_other2_first_noThing2_other2_second------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --thing2NotRun --other2Run --fileNames dropThing2_other2_start11.root noThing2_other2.root || die "cmsRun ${test}cfg.py --thing2NotRun --other2Run --fileNames dropThing2_other2_start11.root noThing2_other2.root" $?

echo ${test}test_noThing2_other2_first_dropThing2_other2_second------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10 --other2Run --thing2Dropped --fileNames noThing2_other2.root dropThing2_other2_start11.root || die "cmsRun ${test}cfg.py --other2Run --thing2Dropped --fileNames noThing2_other2 dropThing2_other2_start11.root" $?

echo ${test}prod_dropThing2_other2_depThing1_start11 ------------------------------------------------------------

#if thing2 is dropped and we still run other2, then other2 will read from thing1 instead thereby changing the provenance
echo ${test}prod_dropThing2_other2_depThing1_start11------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --startEvent=11 --dropThing2 --thing2DependsOnThing1 --addAndStoreOther2  --fileName='dropThing2_other2_depThing1_start11.root' || die "cmsRun ${test}prod_cfg.py --startEvent=11 --dropThing2 --thing2DependsOnThing1 --addAndStoreOther2" $?

#branch structures are the same but in one other2 depends on thing1 while in the other file other2 depends on thing2
echo ${test}test_dropThing2_other2_first_noThing2_other2_second------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10  --thing2NotRun --other2Run --fileNames dropThing2_other2_depThing1_start11.root noThing2_other2.root || die "cmsRun ${test}cfg.py --thing2NotRun --other2Run --fileNames dropThing2_other2_depThing1_start11.root noThing2_other2.root" $?

echo ${test}test_noThing2_other2_first_dropThing2_other2_second------------------------------------------------------------
cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --nEventsToFail=10 --other2Run --thing2Dropped --thing2DependsOnThing1 --fileNames noThing2_other2.root dropThing2_other2_depThing1_start11.root || die "cmsRun ${test}cfg.py --other2Run --thing2Dropped --fileNames noThing2_other2 dropThing2_other2_depThing1_start11.root" $?




exit 0

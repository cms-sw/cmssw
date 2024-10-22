#!/bin/sh


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

${SCRAM_TEST_PATH}/RefTest_a.sh || die 'Failed to create file' $?
root -b -n -q ${SCRAM_TEST_PATH}/event_looping_cint.C || die 'Failed in event_looping_cint.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/event_looping_consumes_cint.C || die 'Failed in event_looping_cint.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/chainevent_looping_cint.C || die 'Failed in chainevent_looping_cint.C' $?
python3 ${SCRAM_TEST_PATH}/chainEvent_python.py || die 'Failed in chainEvent_python.py' $?
#root -b -n -q ${SCRAM_TEST_PATH}/autoload_with_std.C || die 'Failed in autoload_with_std.C' $?
#root -b -n -q ${SCRAM_TEST_PATH}/autoload_with_missing_std.C || die 'Failed in autoload_with_missing_std.C' $?

${SCRAM_TEST_PATH}/MergeTest.sh || die 'Failed to create file' $?
root -b -n -q ${SCRAM_TEST_PATH}/runlumi_looping_cint.C || die 'Failed in runlumi_looping_cint.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/productid_cint.C || die 'Failed in productid_cint.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/triggernames_cint.C || die 'Failed in triggernames_cint.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/triggernames_multi_cint.C || die 'Failed in triggernames_multi_cint.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/triggerResultsByName_cint.C || die 'Failed in triggerResultsByName_cint.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/triggerResultsByName_multi_cint.C || die 'Failed in triggerResultsByName_multi_cint.C' $?

${SCRAM_TEST_PATH}/VIPTest.sh || die 'Failed to create file' $?
root -b -n -q ${SCRAM_TEST_PATH}/vector_int_cint.C || die 'Failed in vector_int_cint.C' $?

python3 ${SCRAM_TEST_PATH}/pyroot_handle_reuse.py || die 'Failed in pyroot_handle_reuse.py' $?
python3 ${SCRAM_TEST_PATH}/pyroot_multichain.py inputFiles=file:prodmerge.root secondaryInputFiles=file:prod1.root,file:prod2.root || die 'Failed in pyroot_multichain.py (non-empty files)' $?
python3 ${SCRAM_TEST_PATH}/pyroot_multichain.py inputFiles=file:empty_a.root secondaryInputFiles=file:good_a.root  || die 'Failed in pyroot_multichain.py (empty file)' $?

#NOTE: ROOT has a bug which keeps the AssociationVector from running its ioread rule and therefore it never clears its cache
#test AssociationVector reading
#rm -f ${SCRAM_TEST_PATH}/avtester.root
#rm -f avtester.root
#cmsRun ${SCRAM_TEST_PATH}/make_associationvector_file_cfg.py  || die "cmsRun make_associationvector_file_cfg.py " $?
#python ${SCRAM_TEST_PATH}/pyroot_read_associationvector.py || die 'Failed in pyroot_read_associationvector.py'

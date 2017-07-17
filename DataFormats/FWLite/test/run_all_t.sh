#!/bin/sh


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

${LOCAL_TEST_DIR}/RefTest_a.sh || die 'Failed to create file' $?
root -b -n -q ${LOCAL_TEST_DIR}/event_looping_cint.C || die 'Failed in event_looping_cint.C' $?
root -b -n -q ${LOCAL_TEST_DIR}/chainevent_looping_cint.C || die 'Failed in chainevent_looping_cint.C' $?
python ${LOCAL_TEST_DIR}/chainEvent_python.py || die 'Failed in chainEvent_python.py' $?
#root -b -n -q ${LOCAL_TEST_DIR}/autoload_with_std.C || die 'Failed in autoload_with_std.C' $?
#root -b -n -q ${LOCAL_TEST_DIR}/autoload_with_missing_std.C || die 'Failed in autoload_with_missing_std.C' $?

${LOCAL_TEST_DIR}/MergeTest.sh || die 'Failed to create file' $?
root -b -n -q ${LOCAL_TEST_DIR}/runlumi_looping_cint.C || die 'Failed in runlumi_looping_cint.C' $?
root -b -n -q ${LOCAL_TEST_DIR}/productid_cint.C || die 'Failed in productid_cint.C' $?
root -b -n -q ${LOCAL_TEST_DIR}/triggernames_cint.C || die 'Failed in triggernames_cint.C' $?
root -b -n -q ${LOCAL_TEST_DIR}/triggernames_multi_cint.C || die 'Failed in triggernames_multi_cint.C' $?
root -b -n -q ${LOCAL_TEST_DIR}/triggerResultsByName_cint.C || die 'Failed in triggerResultsByName_cint.C' $?
root -b -n -q ${LOCAL_TEST_DIR}/triggerResultsByName_multi_cint.C || die 'Failed in triggerResultsByName_multi_cint.C' $?

${LOCAL_TEST_DIR}/VIPTest.sh || die 'Failed to create file' $?
root -b -n -q ${LOCAL_TEST_DIR}/vector_int_cint.C || die 'Failed in vector_int_cint.C' $?

python ${LOCAL_TEST_DIR}/pyroot_handle_reuse.py || die 'Failed in pyroot_handle_reuse.py' $?
python ${LOCAL_TEST_DIR}/pyroot_multichain.py inputFiles=file:prodmerge.root secondaryInputFiles=file:prod1.root,file:prod2.root || die 'Failed in pyroot_multichain.py (non-empty files)' $?
python ${LOCAL_TEST_DIR}/pyroot_multichain.py inputFiles=file:empty_a.root secondaryInputFiles=file:good_a.root  || die 'Failed in pyroot_multichain.py (empty file)' $?

#NOTE: ROOT has a bug which keeps the AssociationVector from running its ioread rule and therefore it never clears its cache
#test AssociationVector reading
#rm -f ${LOCAL_TEST_DIR}/avtester.root
#rm -f avtester.root
#cmsRun ${LOCAL_TEST_DIR}/make_associationvector_file_cfg.py  || die "cmsRun make_associationvector_file_cfg.py " $?
#python ${LOCAL_TEST_DIR}/pyroot_read_associationvector.py || die 'Failed in pyroot_read_associationvector.py'

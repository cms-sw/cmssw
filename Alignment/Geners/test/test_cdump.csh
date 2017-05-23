#!/bin/tcsh

set cdump_cmd=${CMSSW_BASE}/test/${SCRAM_ARCH}/cdump
foreach dir (${CMSSW_BASE} ${CMSSW_RELEASE_BASE} ${CMSSW_FULL_RELEASE_BASE})
  if ( -e ${dir}/test/${SCRAM_ARCH}/cdump ) then
    set cdump_cmd=${dir}/test/${SCRAM_ARCH}/cdump
    break
  endif
end
rm -f $LOCAL_TMP_DIR/cdump.out
${cdump_cmd} -f $LOCAL_TEST_DIR/archive.gsbmf >& $LOCAL_TMP_DIR/cdump.out
diff $LOCAL_TMP_DIR/cdump.out $LOCAL_TEST_DIR/cdump.ref >& /dev/null
if ($status) then
    echo "!!!! Catalog dump regression test FAILED"
else
    echo "**** Catalog dump regression test passed"
endif
rm -f $LOCAL_TMP_DIR/cdump.out

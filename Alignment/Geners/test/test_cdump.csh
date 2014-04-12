#!/bin/tcsh

rm -f $LOCAL_TMP_DIR/cdump.out
$LOCAL_TEST_BIN/cdump -f $LOCAL_TEST_DIR/archive.gsbmf >& $LOCAL_TMP_DIR/cdump.out
diff $LOCAL_TMP_DIR/cdump.out $LOCAL_TEST_DIR/cdump.ref >& /dev/null
if ($status) then
    echo "!!!! Catalog dump regression test FAILED"
else
    echo "**** Catalog dump regression test passed"
endif
rm -f $LOCAL_TMP_DIR/cdump.out

#!/bin/bash
echo "===== Running test of cms_adler32 ======"

expect="ce0bf49e"
results=`cms_adler32 $LOCAL_TEST_DIR/../bin/adler32.c | awk '{print $1}'`

if [ "$results" != "$expect" ]; then
    echo "result: " $results
    echo "expect: " $expect
    echo ">>>>> test of adler32 failed <<<<<"
    exit 1
fi


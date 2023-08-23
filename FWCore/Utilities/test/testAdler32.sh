#!/bin/bash
echo "===== Running test of cms_adler32 ======"

expect="fd466a74"
results=`cms_adler32 ${SCRAM_TEST_PATH}/doNotModify.txt | awk '{print $1}'`

if [ "$results" != "$expect" ]; then
    echo "result: " $results
    echo "expect: " $expect
    echo ">>>>> test of adler32 failed <<<<<"
    exit 1
fi


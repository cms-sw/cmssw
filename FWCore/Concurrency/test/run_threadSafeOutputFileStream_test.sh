#!/bin/bash

# The purpose of this test is to ensure that the entries inserted into
# the ThreadSafeOutputFileStream object are correctly ordered per
# thread.  This involves creating multiple threads all writing to the
# same output file, and then greping the output per thread to make
# sure the order is correct.
#
# The order should be (after greping per thread index):
#
#   Thread index: 0 Entry: 0
#   Thread index: 0 Entry: 1
#   Thread index: 0 Entry: 2
#   Thread index: 0 Entry: 3
#
#   Thread index: 1 Entry: 0
#   Thread index: 1 Entry: 1
#   Thread index: 1 Entry: 2
#   Thread index: 1 Entry: 3
#
#   Thread index: 2 Entry: 0
#   Thread index: 2 Entry: 1
#   Thread index: 2 Entry: 2
#   Thread index: 2 Entry: 3
#
# Since matching grep results to strings can be tricky due to
# linebreaks, I replaced the line breaks with semicolons, and compare
# accordingly.

function die { echo Failure $1: status $2 ; exit $2 ; }

test_threadSafeOutputFileStream || die "test_ThreadSafeOutputFileStream" $?
for thindex in 0 1 2;
do
    comparison="Thread index: ${thindex} Entry: 0;Thread index: ${thindex} Entry: 1;Thread index: ${thindex} Entry: 2;Thread index: ${thindex} Entry: 3;"
    [[ "$( grep "Thread index: ${thindex}" thread_safe_ofstream_test.txt | tr \\n \; )" == "${comparison}" ]] || die "test_ThreadSafeOutputFileStream_output" $?
done

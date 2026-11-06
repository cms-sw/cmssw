#! /bin/bash

# run the streamer tests with the given compressor and compression level, and stop at the first failure
function run_test() {
  TEST_COMPRESSION_ALGO="$1" TEST_COMPRESSION_LEVEL="$2" $(dirname $0)/RunSimple_NewStreamer.sh || { STATUS=$?; echo "$1 compression level $2 failed: status $STATUS"; exit $STATUS; }
}

# ZLIB level 1
run_test ZLIB 1

#! /bin/bash

# run the streamer tests with the given compressor and compression level, and stop at the first failure
function run_test() {
  TEST_COMPRESSION_ALGO="$1" TEST_COMPRESSION_LEVEL="$2" $(dirname $0)/RunSimple_NewStreamer.sh || { STATUS=$?; echo "$1 compression level $2 failed: status $STATUS"; exit $STATUS; }
}

# LZ4 fast compression
run_test LZ4 1

# LZ4 mid-level
run_test LZ4 2

# LZ4 high compression (HC)
run_test LZ4 4

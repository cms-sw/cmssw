#!/bin/bash
export TEST_COMPRESSION_ALGO="LZ4"
export TEST_COMPRESSION_LEVEL=9
$(dirname $0)/RunSimple_NewStreamer.sh

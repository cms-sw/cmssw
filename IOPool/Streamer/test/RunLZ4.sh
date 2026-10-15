#!/bin/bash
export TEST_COMPRESSION_ALGO="LZ4"
$(dirname $0)/RunSimple_NewStreamer.sh

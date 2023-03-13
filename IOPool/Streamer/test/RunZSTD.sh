#!/bin/bash
export TEST_COMPRESSION_ALGO="ZSTD" 
$(dirname $0)/RunSimple_NewStreamer.sh

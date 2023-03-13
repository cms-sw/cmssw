#!/bin/bash
export TEST_COMPRESSION_ALGO="ZLIB" 
$(dirname $0)/RunSimple_NewStreamer.sh

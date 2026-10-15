#!/bin/bash

# LZ4 fast compression
export TEST_COMPRESSION_ALGO="LZ4"
export TEST_COMPRESSION_LEVEL=1
$(dirname $0)/RunSimple_NewStreamer.sh

# LZ4 high compression (HC)
export TEST_COMPRESSION_ALGO="LZ4"
export TEST_COMPRESSION_LEVEL=4
$(dirname $0)/RunSimple_NewStreamer.sh

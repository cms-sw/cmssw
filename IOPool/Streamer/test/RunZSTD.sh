#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export TEST_COMPRESSION_ALGO="ZSTD" 
exec ${SCRIPTDIR}/RunSimple_NewStreamer.sh

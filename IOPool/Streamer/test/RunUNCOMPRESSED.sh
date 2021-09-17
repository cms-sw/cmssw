#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export TEST_COMPRESSION_ALGO="UNCOMPRESSED" 
exec ${SCRIPTDIR}/RunSimple_NewStreamer.sh

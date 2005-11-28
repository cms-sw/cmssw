#!/bin/bash

source setup.sh
source stats.sh

SCRAM_ARCH=slc3_ia32_gcc323
CMSSWVERSION=CMSSW_0_2_0_pre7
THISDIR=`pwd`
CMSSWVERSION=CMSSW_0_2_0_pre7 #change to the CMSSW version for testing
export CVSROOT=:kserver:cmscvs.cern.ch:/cvs_server/repositories/CMSSW
export SCRAM_ARCH=slc3_ia32_gcc323
READER=${CMSSWVERSION}/test/${SCRAM_ARCH}/OCCIPedReader
LIMIT=0 # Limit to N object, 0 for no limit
NUMTESTS=5

bootstrap_cmssw ${CMSSWVERSION}
echo "[---JOB LOG---] bootstrap_cmssw status $?"
setup_tns
echo  "[---JOB LOG---] setup_tns status $?"

echo "[---JOB LOG---] Running ${READER} Mode 0 Limit ${LIMIT}"
runx "${READER} 0 ${LIMIT}" 5
echo "[---JOB LOG---] Running ${READER} Mode 1 Limit ${LIMIT}"
runx "${READER} 1 ${LIMIT}" 5

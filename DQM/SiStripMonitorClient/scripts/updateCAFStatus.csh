#!/bin/tcsh
set OLD_DIR=${PWD}
cd ~cctrack/DQM/${CMSSW_VERSION}/src/DQM/SiStripMonitorClient/scripts/${1}/
crab -status -c crab${1}
crab -get    -c crab${1}
crab -status -c crab${1}
echo
set CASTOR_DIR=/castor/cern.ch/user/c/cctrack/DQM
echo "Found in ${CASTOR_DIR}:"
rfdir ${CASTOR_DIR} | grep ${1}
echo
cd $OLD_DIR

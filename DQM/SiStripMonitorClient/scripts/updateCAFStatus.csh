#!/bin/tcsh
set OLD_DIR=${PWD}
cd ~cctrack/DQM/${CMSSW_VERSION}/src/DQM/SiStripMonitorClient/scripts/${1}/
crab -status -c crab${1}
crab -get    -c crab${1}
crab -status -c crab${1}
cd $OLD_DIR

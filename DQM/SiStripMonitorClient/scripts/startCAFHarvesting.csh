#!/bin/tcsh
set OLD_DIR=${PWD}
cd ~cctrack/DQM/${CMSSW_VERSION}/src/DQM/SiStripMonitorClient/scripts/${1}
bsub -q cmscaf SiStripCAFHarvest.job
tar -czvf crab${1}.tar.gz crab${1}
rm -r crab${1}
cd ${OLD_DIR}

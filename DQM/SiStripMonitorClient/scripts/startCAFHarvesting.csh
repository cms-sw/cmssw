#!/bin/tcsh
set OLD_DIR=${PWD}
cd ~cctrack/DQM/${CMSSW_VERSION}/src/DQM/SiStripMonitorClient/scripts/${1}
bsub -q cmscaf SiStripCAFHarvest.job
if ( ! -e crab${1}.tar.gz ) then
  tar -czvf crab${1}.tar.gz crab${1}
  rm -r crab${1}
endif
cd ${OLD_DIR}

#!/bin/tcsh
echo $HOME
set casa = `pwd`
pwd 
ls -la 
echo $HOME
cd $HOME/scratch0/CMSSW/Offline/CMSSW_3_11_0/src/DQM/RPCMonitorDigi/test
eval `scramv1 runtime -csh`
cp rpcdqm_cfg.py $casa
cd $casa
cmsRun -p rpcdqm_cfg.py

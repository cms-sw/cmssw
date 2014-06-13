cmssw
=====


cmsrel CMSSW_6_2_0_SLHC6

cd CMSSW_6_2_0_SLHC6/src

cmsenv

git cms-init

git checkout -b DQM_TRK_Upgrade_SLHC6

git remote add idebruyn-cmssw git@github.com:idebruyn/cmssw.git

git fetch idebruyn-cmssw

git merge idebruyn-cmssw/DQM_TRK_Upgrade_SLHC6



git cms-addpkg DQM/OutTrackerMonitorClient

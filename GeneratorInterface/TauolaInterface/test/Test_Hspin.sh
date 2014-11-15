#! /bin/bash

cmsRun H130GGgluonfusiontaupinu_Phi0_8TeV_Tauola_TauSpinner_Example_cfi.py
cmsDriver.py step3 -s HARVESTING:genHarvesting --harvesting AtJobEnd --conditions auto:mc  --mc --filein file:step1.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root  H130GGgluonfusiontaupinu_Phi0.root

cmsRun H130GGgluonfusiontaupinu_PhiHalfPi_8TeV_Tauola_TauSpinner_Example_cfi.py
cmsDriver.py step3 -s HARVESTING:genHarvesting --harvesting AtJobEnd --conditions auto:mc  --mc --filein file:step1.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root H130GGgluonfusiontaupinu_PhiHalfPi.root

cmsRun H130GGgluonfusiontaupinu_PhiQuarterPi_8TeV_Tauola_TauSpinner_Example_cfi.py
cmsDriver.py step3 -s HARVESTING:genHarvesting --harvesting AtJobEnd --conditions auto:mc  --mc --filein file:step1.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root H130GGgluonfusiontaupinu_PhiQuarterPi.root

cmsRun H130GGgluonfusiontaurhonu_Phi0_8TeV_Tauola_TauSpinner_Example_cfi.py
cmsDriver.py step3 -s HARVESTING:genHarvesting --harvesting AtJobEnd --conditions auto:mc  --mc --filein file:step1.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root  H130GGgluonfusiontaurhonu_Phi0.root

cmsRun H130GGgluonfusiontaurhonu_PhiHalfPi_8TeV_Tauola_TauSpinner_Example_cfi.py
cmsDriver.py step3 -s HARVESTING:genHarvesting --harvesting AtJobEnd --conditions auto:mc  --mc --filein file:step1.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root H130GGgluonfusiontaurhonu_PhiHalfPi.root

cmsRun H130GGgluonfusiontaurhonu_PhiQuarterPi_8TeV_Tauola_TauSpinner_Example_cfi.py
cmsDriver.py step3 -s HARVESTING:genHarvesting --harvesting AtJobEnd --conditions auto:mc  --mc --filein file:step1.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root H130GGgluonfusiontaurhonu_PhiQuarterPi.root

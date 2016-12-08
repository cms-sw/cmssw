#1. produce the DQM histograms needed as input to the calibration. This eventually will run in the AlCaSplitting step @ Tier0 which produces the so called ALCAPROMPT dataset. This step runs in parallel jobs.
cmsDriver.py step3 --datatier ALCARECO,DQM \
--conditions MCRUN2_75_V1 \
-s ALCA:PromptCalibProdSiStripGains \
--eventcontent ALCARECO,DQM -n 100 \
--dasquery='file dataset=/RelValMinBias_13/CMSSW_7_5_0_pre4-MCRUN2_75_V1-v1/GEN-SIM-RECO' \
--fileout file:step3.root --no_exec

#dasquerry can be replaced by a dasquerry eventually
#--dbsquery="find file where dataset = /MinimumBias/Run2012C-SiStripCalMinBias-v2/ALCARECO and run=200190" \


#2. produce the sqlite and the DQM file to be uplaoded on the GUI. This the so called AlCaHarvesting step @ Tier0 and is actually a unique job running on all the files of a given run.
cmsDriver.py step4  --data  --conditions auto:com10 \
--scenario pp \
-s ALCAHARVEST:SiStripGains \
--filein file:PromptCalibProdSiStripGains.root -n -1 --no_exec


#'/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/FAFF2948-4EDF-E111-97FB-BCAEC518FF44.root'



cmsDriver.py step1 --conditions auto:com10 -s RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForExpress+SiStripCalMinBias+SiStripPCLHistos,DQM,ENDJOB --process RECO --data --eventcontent ALCARECO,DQM --scenario pp --datatier ALCARECO,DQM --customise Configuration/DataProcessing/RecoTLR.customiseExpress -n 100 --dasquery="file dataset = /MinimumBias/Run2012C-v1/RAW run=200190" --fileout file:step1.root

cmsDriver.py step2 --datatier ALCARECO --conditions auto:com10 -s ALCA:PromptCalibProdSiStripGains --eventcontent ALCARECO -n 100 --filein file:step1.root



cmsDriver.py step1 --conditions auto:com10 -s RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForExpress+SiStripCalMinBias+SiStripPCLHistos,DQM,ENDJOB --process RECO --data --eventcontent ALCARECO,DQM --scenario pp --datatier ALCARECO,DQM --customise Configuration/DataProcessing/RecoTLR.customiseExpress -n 100 --dasquery="file dataset = /MinimumBias/Run2012C-v1/RAW run=200190" --fileout file:step1.root
cmsDriver.py step2 --datatier ALCARECO --conditions auto:com10 -s ALCA:PromptCalibProdSiStripGains --eventcontent ALCARECO -n 100 --filein file:step1.root

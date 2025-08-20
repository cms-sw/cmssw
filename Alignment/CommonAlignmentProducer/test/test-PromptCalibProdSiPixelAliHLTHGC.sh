#!/bin/bash -ex

cmsDriver.py testReAlCaHLT \
	     -s ALCA:TkAlHLTTracks+TkAlHLTTracksZMuMu  \
	     --conditions 140X_dataRun3_Express_v3 \
	     --scenario pp \
	     --data \
	     --era Run3_2024 \
	     --datatier ALCARECO \
	     --eventcontent ALCARECO \
	     --processName=ReAlCa \
	     -n 100000 \
	     --dasquery='file dataset=/HLTMonitor/Run2024I-Express-v2/FEVTHLTALL site=T2_CH_CERN' \
	     --nThreads 4 >& ReAlCa.log

cmsDriver.py testReAlCaHLTHGComb \
	     -s ALCA:PromptCalibProdSiPixelAliHLTHGC \
	     --conditions 140X_dataRun3_Express_v3 \
	     --scenario pp \
	     --data \
	     --era Run3_2024 \
	     --datatier ALCARECO \
	     --eventcontent ALCARECO \
	     --processName=ReAlCaHLTHGC \
	     -n -1 \
	     --filein file:TkAlHLTTracks.root \
	     --customise Alignment/CommonAlignmentProducer/customizeLSNumberFilterForRelVals.doNotFilterLS \
	     --customise_commands='process.ALCARECOTkAlZMuMuFilterForSiPixelAliHLT.throw = False;process.ALCARECOTkAlMinBiasFilterForSiPixelAliHLTHG.TriggerResultsTag = "TriggerResults::ReAlCa"' \
	     --triggerResultsProcess ReAlCa \
	     --nThreads 4 >& HLTHGComb_1.log
	     
rm -rf *.dat
mv PromptCalibProdSiPixelAliHLTHGC.root PromptCalibProdSiPixelAliHLTHGC_0.root

cmsDriver.py testReAlCaHLTHGComb \
	     -s ALCA:PromptCalibProdSiPixelAliHLTHGC \
	     --conditions 140X_dataRun3_Express_v3 \
	     --scenario pp \
	     --data \
	     --era Run3_2024 \
	     --datatier ALCARECO \
	     --eventcontent ALCARECO \
	     --processName=ReAlCaHLTHGC \
	     -n -1 \
	     --filein file:TkAlHLTTracksZMuMu.root  \
	     --customise Alignment/CommonAlignmentProducer/customizeLSNumberFilterForRelVals.doNotFilterLS \
	     --customise_commands='process.ALCARECOTkAlZMuMuFilterForSiPixelAliHLT.TriggerResultsTag = "TriggerResults::ReAlCa";process.ALCARECOTkAlMinBiasFilterForSiPixelAliHLTHG.throw = False' \
	     --triggerResultsProcess ReAlCa \
	     --nThreads 4 >&  HLTHGComb_2.log
	     
mv PromptCalibProdSiPixelAliHLTHGC.root PromptCalibProdSiPixelAliHLTHGC_1.root

cmsDriver.py stepHarvest \
	     -s ALCAHARVEST:SiPixelAliHLTHGCombined \
	     --conditions 140X_dataRun3_Express_v3 \
	     --scenario pp \
	     --data \
	     --era Run3_2024 \
	     -n -1 \
	     --filein file:PromptCalibProdSiPixelAliHLTHGC_0.root,file:PromptCalibProdSiPixelAliHLTHGC_1.root >& Harvesting.log

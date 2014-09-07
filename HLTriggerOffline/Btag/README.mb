---------------------- DESCRIPTION ------------------------------------------------- 

HLTriggerOffline/Btag (avilable on https://github.com/silviodonato/cmssw/tree/hlt-validation-2/HLTriggerOffline/Btag)
This package is used by HLT b-tag group for the CMSSW release validation.

---------------------- INSTALLATION ------------------------------------------------- 

cd CMSSW_7_2_0_pre4
cd src/
cmsenv
git cms-merge-topic silviodonato:hlt-validation-2
scram b -j4

---------------------- USE ------------------------------------------------- 

To produce the DQM plots of a CMSSW release do:
	1. Take a file containing the HLT b-tag informations reconstructed with the CMSSW release that you want to validate  (*)
	2. cd HLTriggerOffline/Btag/test
	3. Modify config.ini as you need (files, cmsswver, ...)
	4. cmsRun hltHarvesting_cff.py >& logHarvesting &
	5. Now you have a DQM_V0001_R000000001__CMSSW_X_Y_Z__RelVal__TrigVal.root file containing the HLT btag DQM plots (/DQMData/Run 1/HLT/Run summary/Btag or Vertex).

To compare two DQM file:
	1. cd HLTriggerOffline/Btag/test
	2. Modify compareDQM.py as you need (at least the input files)
	3. ./compareDQM.py
	4. Now you have a 'plots' folders containing the comparison between the two CMSSW release
	

----------------------------------------------------------------------------------------------------- 

(*)
These b-tag informations will be save in the standard HLTDEBUG format (e.g. '/RelValTTbar_13/CMSSW_7_***-POSTLS172_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')

To produce and to save them by your own:

1. Download the last btag-related HLT paths (choose a MC RAW input file, run the trigger in the open mode, choose the max-events ...):

hltGetConfiguration /dev/CMSSW_7_1_2/HLT --globaltag auto:startup_GRun --input file:/afs/cern.ch/work/n/ntsirova/public/hlt_dev/ttbar-13tev-1file.root --output full --paths HLT_PFMET120_NoiseCleaned_BTagCSV07_v1,HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1,HLT_BTagCSV07_v1 --open --max-events 100 > btag-pathes.py

2. (optional) Replace the output modules in btag-pathes.py, in order to keep only the useful informations:

process.outp1=cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string('outputFULL.root'),
        outputCommands = cms.untracked.vstring(
        'drop *',
        'keep recoVertexs_*_*_*',
        'keep recoCaloJets_*_*_*',
        'keep recoPFJets_*_*_*',
        'keep recoTracksRefsrecoJTATagInforecoIPTagInforecoVertexrecoTemplatedSecondaryVertexTagInfos_*_*_*',
        'keep recoJetedmRefToBaseProdTofloatsAssociationVector_*_*_*',

        'keep *_TriggerResults_*_*',
        'keep *_genParticles_*_*',        
        'keep SimVertexs_g4SimHits_*_*',
         )
)  
process.out = cms.EndPath( process.outp1 )

You can check an example in: /afs/cern.ch/user/s/sdonato/AFSwork/public/btag-pathes.py

3. Launch the job:

cmsRun btag-pathes.py >& logHLTbtag &

import FWCore.ParameterSet.Config as cms
process = cms.Process("HLTBTAG")

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("HLTriggerOffline.Btag.HltBtagValidation_cff")
#process.load("HLTriggerOffline.Btag.HltBtagValidationFastSim_cff")
process.load("HLTriggerOffline.Btag.HltBtagPostValidation_cff")

process.DQM_BTag = cms.Path(    process.hltbtagValidationSequence + process.HltBTagPostVal + process.dqmSaver)


process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring(
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_1_7/RelValTTbar_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_HEmiss_v1-v1/10000/FE7C783A-9F7F-E811-B920-0CC47A7452D8.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_1_7/RelValTTbar_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_HEmiss_v1-v1/10000/6247ABC0-9E7F-E811-BEF3-0CC47A7C354A.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_1_7/RelValTTbar_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_HEmiss_v1-v1/10000/52F6F6B2-9E7F-E811-8EDD-0CC47A7C356A.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_1_7/RelValTTbar_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_HEmiss_v1-v1/10000/BADE38B2-9E7F-E811-9C9C-0CC47A78A426.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_1_7/RelValTTbar_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_HEmiss_v1-v1/10000/6A31CAC8-9E7F-E811-ABB7-0025905A48FC.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_1_7/RelValTTbar_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_HEmiss_v1-v1/10000/F8348CC5-9E7F-E811-8198-0025905A6080.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_1_7/RelValTTbar_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_HEmiss_v1-v1/10000/3295A9B8-9E7F-E811-A7EF-0025905A48E4.root'

)
#	fileNames = cms.untracked.vstring("file:RelVal750pre3.root")
)



#Settings equivalent to 'RelVal' convention:
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.dqmSaver.workflow = "/test/RelVal/TrigVal"
process.DQMStore.collateHistograms = False
process.DQMStore.verbose=0
process.options = cms.untracked.PSet(
	wantSummary	= cms.untracked.bool( True ),
	fileMode	= cms.untracked.string('FULLMERGE'),
	SkipEvent	= cms.untracked.vstring('ProductNotFound')
)


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
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW/102X_upgrade2018_realistic_v7-v1/10000/103778EC-A27B-E811-8D04-0CC47A4D76AA.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW/102X_upgrade2018_realistic_v7-v1/10000/6C1229C7-A37B-E811-892F-0025905B860C.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW/102X_upgrade2018_realistic_v7-v1/10000/1E0840F2-A57B-E811-999A-0CC47A4D76AA.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW/102X_upgrade2018_realistic_v7-v1/10000/80244DCF-A67B-E811-AA38-0CC47A78A3EE.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW/102X_upgrade2018_realistic_v7-v1/10000/B2F774FD-A57B-E811-B7DE-0025905B860C.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW/102X_upgrade2018_realistic_v7-v1/10000/DCA784B2-A77B-E811-8D49-0CC47A4D7698.root',
'root://cms-xrd-global.cern.ch//store/relval/CMSSW_10_2_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW/102X_upgrade2018_realistic_v7-v1/10000/B2CC07B7-A77B-E811-AC18-0025905B8562.root'

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


import FWCore.ParameterSet.Config as cms
process = cms.Process("HLTBTAG")

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("HLTriggerOffline.Btag.HltBtagValidation_cff")
process.load("HLTriggerOffline.Btag.HltBtagValidationFastSim_cff")
process.load("HLTriggerOffline.Btag.HltBtagPostValidation_cff")

process.DQM_BTag = cms.Path(    process.hltbtagValidationSequence + process.HltBTagPostVal + process.dqmSaver)


process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring("root://xrootd.ba.infn.it///store/relval/CMSSW_7_5_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_75X_mcRun2_asymptotic_v1-v1/00000/00DE8AFD-1D1C-E511-988E-0025905A612C.root")
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


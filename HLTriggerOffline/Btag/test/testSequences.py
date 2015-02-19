import FWCore.ParameterSet.Config as cms
process = cms.Process("HLTBTAG")

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("HLTriggerOffline.Btag.HltBtagValidation_cff")
process.load("HLTriggerOffline.Btag.HltBtagValidationFastSim_cff")
process.load("HLTriggerOffline.Btag.HltBtagPostValidation_cff")

process.DQM_BTag = cms.Path(    process.hltbtagValidationSequence + process.HltBTagPostVal)


process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring("root://xrootd.ba.infn.it///store/relval/CMSSW_7_2_0_pre8/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS172_V15-v1/00000/8A425442-2A50-E411-A51B-0025905A60B2.root")
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


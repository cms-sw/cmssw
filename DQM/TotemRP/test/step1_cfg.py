import FWCore.ParameterSet.Config as cms

process = cms.Process('RECODQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')  #for MC

# load DQM frame work
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

# include TOTEM reconstruction chain
process.load("reco_chain_cfi")

# specify number of events to select
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

# TOTEM RP DQM module
process.TotemRPDQMSource = cms.EDAnalyzer("TotemRPDQMSource",
    tagStripDigi = cms.InputTag("TotemRawToDigi"),
	tagDigiCluster = cms.InputTag("RPClustProd"),
	tagRecoHit = cms.InputTag("RPRecoHitProd"),
	tagPatternColl = cms.InputTag("NonParallelTrackFinder"),
	tagTrackColl = cms.InputTag("RPSingleTrackCandCollFit"),
	tagTrackCandColl = cms.InputTag("NonParallelTrackFinder"),
	tagMultiTrackColl = cms.InputTag(""),

    buildCorrelationPlots = cms.untracked.bool(False),
    correlationPlotsLimit = cms.untracked.uint32(50),
	correlationPlotsFilter = cms.untracked.string("default=0,1")
)

# DQM output
process.DQMOutput = cms.OutputModule("DQMRootOutputModule",
  fileName = cms.untracked.string("OUT_step1.root")
)

# execution schedule
process.dqm_offline_step = cms.Path(process.TotemRPDQMSource)
process.dqm_output_step = cms.EndPath(process.DQMOutput)

process.schedule = cms.Schedule(
    process.reco_step,
    process.dqm_offline_step,
    process.dqm_output_step
)

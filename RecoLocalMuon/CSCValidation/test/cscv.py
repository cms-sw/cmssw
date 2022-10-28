# Test CSCValidation in 12_x - Tim Cox - 15.07.2022
# from SingleMuon RAW

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("TEST", Run3)

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# 2022
##process.GlobalTag.globaltag = 'auto:phase1_2022_realistic' ## FAILS IN 12_3_6
# 12_3_6 Prompt
process.GlobalTag.globaltag = '123X_dataRun3_Prompt_v12'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 5000 ) )  ## UP TO 5K EVENTS
process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
##     "root://eoscms.cern.ch://eos/cms/tier0/store/data/Run2022B/SingleMuon/RAW/v1/000/355/208/00000/3004d340-6e3d-4223-9eb2-7d50a65bfe57.root"
  "/store/data/Run2022B/SingleMuon/RAW-RECO/ZMu-PromptReco-v1/000/355/558/00000/ce02cf0e-e8f7-471c-b533-61b6f124c095.root"
)
)

process.cscValidation = cms.EDAnalyzer("CSCValidation",
    rootFileName = cms.untracked.string('cscv.root'),
    isSimulation = cms.untracked.bool(False),
    writeTreeToFile = cms.untracked.bool(True),
    useDigis = cms.untracked.bool(True),
    detailedAnalysis = cms.untracked.bool(False),
    useTriggerFilter = cms.untracked.bool(False),
    useQualityFilter = cms.untracked.bool(False),
    makeStandalonePlots = cms.untracked.bool(False),
    makeTimeMonitorPlots = cms.untracked.bool(True),
    rawDataTag = cms.InputTag("rawDataCollector"),
    alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
    clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
    corrlctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    compDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
    cscRecHitTag = cms.InputTag("csc2DRecHits"),
    cscSegTag = cms.InputTag("cscSegments"),
    saMuonTag = cms.InputTag("standAloneMuons"),
    l1aTag = cms.InputTag("gtDigis"),
    hltTag = cms.InputTag("TriggerResults::HLT"),
    makeHLTPlots = cms.untracked.bool(True),
    simHitTag = cms.InputTag("g4SimHits", "MuonCSCHits")
)

# From RECO (and FEVT?)
##process.p = cms.Path(process.cscValidation)

# From RAW
process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscValidation)

# Path and EndPath definitions
##process.raw2digi_step = cms.Path(process.RawToDigi)
##process.reconstruction_step = cms.Path(process.reconstruction)
##process.cscvalidation_step = cms.Path(process.cscValidation)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
##process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.cscvalidation_step,process.endjob_step)
process.schedule = cms.Schedule(process.p,process.endjob_step)

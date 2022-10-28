# Test CSCValidation in 12_x - Tim Cox - 10.07.2022
# from FEVT in Express but unpack and local reco

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
# 12_3_6 Express
process.GlobalTag.globaltag = '123X_dataRun3_Express_v10'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 5000 ) )  ## PROCESS 5K EVENTS
process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        "/store/express/Run2022B/ExpressPhysics/FEVT/Express-v1/000/355/135/00000/0135acb6-75de-4798-9b1f-7fca60fa4029.root"
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

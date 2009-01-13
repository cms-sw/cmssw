import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# if the data file doesn't have reco
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.GlobalTag.globaltag = "IDEAL_30X::All"

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000))

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(False),
    debugVebosity = cms.untracked.uint32(20),
    fileNames = cms.untracked.vstring(#'/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_30X_v1/0005/50E9BA78-E9DD-DD11-8AC9-000423D98B08.root')
'/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/38E34C97-E8DD-DD11-8327-000423D94534.root',
'/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/6EFD547F-E9DD-DD11-B456-000423D99264.root',
'/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/D6EBDF31-41DE-DD11-91F0-000423D952C0.root')
)

process.getpt = cms.EDAnalyzer("MuonSeedParametrization",
    minCSCHitsPerSegment = cms.int32(5),
    recHitLabel = cms.untracked.string("csc2DRecHits"),
    cscSegmentLabel = cms.untracked.string("cscSegments"),
    DebugMuonSeed = cms.bool(False),
    rootFileName = cms.untracked.string('pt_estimate_10.root'),
    EnableDTMeasurement = cms.bool(True),
    dtrecHitLabel = cms.untracked.string('dt1DRecHits'),
    dtSegmentLabel = cms.untracked.string("dt4DSegments"),
    debug = cms.untracked.bool(False),
    simTrackLabel = cms.untracked.string('g4SimHits'),
    simHitLabel = cms.untracked.string('g4SimHits'),
    EnableCSCMeasurement = cms.bool(True)

)


process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.muonCSCDigis+process.muonDTDigis+process.muonRPCDigis + process.muonlocalreco+process.getpt)



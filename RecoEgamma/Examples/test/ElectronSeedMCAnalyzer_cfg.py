import FWCore.ParameterSet.Config as cms

process = cms.Process("readseeds")

process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.L1Reco_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring (
    '/store/relval/CMSSW_3_1_0_pre9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/F8A5EC4A-F64E-DE11-8160-001D09F23A20.root'
    )
)
process.electronSeedAnalysis = cms.EDAnalyzer("ElectronSeedAnalyzer",
    beamSpot = cms.InputTag('offlineBeamSpot'),
    inputCollection = cms.InputTag("ecalDrivenElectronSeeds"),
)

process.mylocalreco =  cms.Sequence(process.trackerlocalreco*process.calolocalreco*process.particleFlowCluster)
process.myglobalreco = cms.Sequence(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks+process.ecalClusters+process.caloTowersRec*process.vertexreco)
process.myelectronseeding = cms.Sequence(process.trackerDrivenElectronSeeds*process.ecalDrivenElectronSeeds*process.electronMergedSeeds)
process.myelectrontracking = cms.Sequence(process.electronCkfTrackCandidates*process.electronGsfTracks)

#process.p = cms.Path(process.RawToDigi*process.mylocalreco*process.myglobalreco*process.myelectronseeding*process.myelectrontracking*process.particleFlowReco*process.pfElectronTranslator*process.gsfElectronSequence*)
process.p = cms.Path(process.RawToDigi*process.mylocalreco*process.myglobalreco*process.myelectronseeding*process.electronSeedAnalysis)


process.GlobalTag.globaltag = 'MC_31X_V3::All'



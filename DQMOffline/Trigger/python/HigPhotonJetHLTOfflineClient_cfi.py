import FWCore.ParameterSet.Config as cms

higPhotonJetHLTOfflineClient = cms.EDAnalyzer(
    "DQMGenericClient",

    subDirs        = cms.untracked.vstring("HLT/xshi"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(), 
    efficiencyProfile = cms.untracked.vstring(),
)





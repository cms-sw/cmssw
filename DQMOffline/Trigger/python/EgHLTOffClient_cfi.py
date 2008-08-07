import FWCore.ParameterSet.Config as cms

egHLTOffDQMClient = cms.EDFilter("EgHLTOfflineClient",
    sourceModuleName = cms.string('egammaHLTDQM'),
    DQMDirName=cms.string("HLT/EgammaHLTOffline_egammaHLTDQM")               
)



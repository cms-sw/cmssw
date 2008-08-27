import FWCore.ParameterSet.Config as cms

egHLTOffDQMClient = cms.EDFilter("EgHLTOfflineClient",
    sourceModuleName = cms.string('egammaHLTDQM'),
    DQMDirName=cms.string("HLTOffline/EgammaHLTOffline_egammaHLTDQM")               
)



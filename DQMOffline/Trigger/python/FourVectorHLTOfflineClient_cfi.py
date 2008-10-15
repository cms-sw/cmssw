import FWCore.ParameterSet.Config as cms

hltFourVectorClient = cms.EDFilter("FourVectorHLTClient",
    hltClientDir = cms.untracked.string('HLTOffline/FourVectorHLTOfflineClient/'),
    hltSourceDir = cms.untracked.string('HLTOffline/FourVectorHLTOfflinehltResults'),
    prescaleLS = cms.untracked.int32(-1),
    prescaleEvt = cms.untracked.int32(1)
)


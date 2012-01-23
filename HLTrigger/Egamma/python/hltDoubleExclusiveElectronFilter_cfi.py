import FWCore.ParameterSet.Config as cms

hltDoubleExclusiveElectronFilter = cms.EDFilter("HLTEgammaDoubleEtPhiFilter",
    etcut1 = cms.double(6.0),
    etcut2 = cms.double(6.0),
    npaircut = cms.int32(1),
    MaxAcop = cms.double(0.6),
    MinEtBalance = cms.double(-1.0),
    MaxEtBalance = cms.double(10.0),
    candTag = cms.InputTag("hltDoubleL1MatchFilter"),
    MinAcop = cms.double(-0.1),
    saveTags = cms.bool( False )
)



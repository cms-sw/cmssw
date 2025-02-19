import FWCore.ParameterSet.Config as cms

#PHOTONS
hltEgammaDoubleEtFilter = cms.EDFilter("HLTEgammaDoubleEtFilter",
    etcut1 = cms.double(30.0),
    candTag = cms.InputTag("hltTrackIsolFilter"),
    etcut2 = cms.double(20.0),
    npaircut = cms.int32(1)
)



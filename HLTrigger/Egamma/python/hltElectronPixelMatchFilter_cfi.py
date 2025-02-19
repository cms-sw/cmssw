import FWCore.ParameterSet.Config as cms

hltElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    candTag = cms.InputTag("hltEgammaHcalIsolFilter"),
    L1IsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    saveTags = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)

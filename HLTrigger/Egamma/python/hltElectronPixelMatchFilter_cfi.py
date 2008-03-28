import FWCore.ParameterSet.Config as cms

hltElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltEgammaHcalIsolFilter")
)



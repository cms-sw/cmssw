import FWCore.ParameterSet.Config as cms

hltZeePMMassFilter = cms.EDFilter("HLTPMMassFilter",
    upperMassCut = cms.double(99999.9), ## effectively infinite

    candTag = cms.InputTag("hltDoubleElectronTrackIsolFilter"),
    nZcandcut = cms.int32(1), ## by candidate we now mean Z candiate

    lowerMassCut = cms.double(54.22)
)



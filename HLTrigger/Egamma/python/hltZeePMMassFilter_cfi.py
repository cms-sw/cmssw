import FWCore.ParameterSet.Config as cms

hltZeePMMassFilter = cms.EDFilter("HLTPMMassFilter",
    lowerMassCut = cms.double(54.22),
    upperMassCut = cms.double(99999.9), ## effectively infinite
    candTag = cms.InputTag("hltDoubleElectronTrackIsolFilter"),
    nZcandcut = cms.int32(1), ## by candidate we now mean Z candiate
    saveTags = cms.bool( False )
)



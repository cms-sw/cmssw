import FWCore.ParameterSet.Config as cms

maxLayer1Electrons = cms.EDFilter("PATElectronMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Electrons")
)



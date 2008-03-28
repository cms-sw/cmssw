import FWCore.ParameterSet.Config as cms

# module to filter on the maximal number of Electrons
maxLayer1Electrons = cms.EDFilter("PATElectronMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Electrons")
)



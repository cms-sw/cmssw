import FWCore.ParameterSet.Config as cms

minLayer1Electrons = cms.EDFilter("PATElectronMinFilter",
    src = cms.InputTag("selectedLayer1Electrons"),
    minNumber = cms.uint32(0)
)



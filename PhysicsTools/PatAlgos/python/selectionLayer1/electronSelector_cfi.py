import FWCore.ParameterSet.Config as cms

selectedLayer1Electrons = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("allLayer1Electrons"),
    cut = cms.string('pt > 10. & abs(eta) < 2.4')
)



import FWCore.ParameterSet.Config as cms

selectedLayer1Taus = cms.EDFilter("PATTauSelector",
    src = cms.InputTag("allLayer1Taus"),
    cut = cms.string('pt > 15. & abs(eta) < 2.4 & emEnergyFraction<0.9 & eOverP>0.5')
)



import FWCore.ParameterSet.Config as cms

selectedLayer1Muons = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("allLayer1Muons"),
    cut = cms.string('pt > 10. & abs(eta) < 2.4')
)



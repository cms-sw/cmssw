import FWCore.ParameterSet.Config as cms

# module to select displacedMuons
#
selectedPatDisplacedMuons = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("patDisplacedMuons"),
    cut = cms.string("")
)



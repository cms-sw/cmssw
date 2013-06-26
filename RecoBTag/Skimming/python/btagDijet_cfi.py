import FWCore.ParameterSet.Config as cms

btagDijet = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeCone5CaloJets"),
    etMin = cms.double(20.0),
    minNumber = cms.uint32(2)
)



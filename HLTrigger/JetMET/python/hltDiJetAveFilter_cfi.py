import FWCore.ParameterSet.Config as cms

hltDiJetAveFilter = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    minEtAve = cms.double(60.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)



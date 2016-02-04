import FWCore.ParameterSet.Config as cms

hltDiJetAveFilter = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTag = cms.untracked.bool( False ),
    minPtAve = cms.double(60.0),
    minDphi = cms.double(0.0),
    minPtJet3 = cms.double(99999.0)
)



import FWCore.ParameterSet.Config as cms

hltPhiJet1Jet2Filter = cms.EDFilter("HLTAcoFilter",
    maxDeltaPhi = cms.double(2.7646),
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTags = cms.bool( False ),
    Acoplanar = cms.string('Jet1Jet2'),
    inputMETTag = cms.InputTag("hlt1MET70"),
    minDeltaPhi = cms.double(0.0),
    minEtJet1 = cms.double(40.0),
    minEtJet2 = cms.double(40.0)
)



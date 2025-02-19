import FWCore.ParameterSet.Config as cms

hltPhiJet2METFilter = cms.EDFilter("HLTAcoFilter",
    maxDeltaPhi = cms.double(3.141593),
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTags = cms.bool( False ),
    Acoplanar = cms.string('Jet2Met'),
    inputMETTag = cms.InputTag("hlt1MET70"),
    minDeltaPhi = cms.double(0.377),
    minEtJet1 = cms.double(50.0),
    minEtJet2 = cms.double(50.0)
)



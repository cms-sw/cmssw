import FWCore.ParameterSet.Config as cms

hltPhiJet1METFilter = cms.EDFilter("HLTAcoFilter",
    maxDeltaPhi = cms.double(2.89),
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTags = cms.bool( False ),
    Acoplanar = cms.string('Jet1Met'),
    inputMETTag = cms.InputTag("hlt1MET70"),
    minDeltaPhi = cms.double(0.0),
    minEtJet1 = cms.double(60.0),
    minEtJet2 = cms.double(-1.0)
)



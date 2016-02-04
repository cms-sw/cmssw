import FWCore.ParameterSet.Config as cms

hltMhtHtFilter = cms.EDFilter("HLTMhtHtFilter",
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTag = cms.untracked.bool( False ),
    minMht = cms.double(100.0),
    minPtJet = cms.vdouble(20.0,30.0),
    minNJet = cms.untracked.int32(3),
    mode = cms.untracked.int32(1),
    etaJet = cms.vdouble(3.0,5.0),
    usePt = cms.bool( True ),
    minPT12 = cms.double(60.0),
    minMeff = cms.double(150.0),
    minHt = cms.double(100.0),
    minMht2Ht = cms.double(0.4)
)



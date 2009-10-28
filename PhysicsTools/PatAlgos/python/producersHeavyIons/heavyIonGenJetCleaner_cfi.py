import FWCore.ParameterSet.Config as cms

heavyIonCleanedGenJets = cms.EDProducer('HiGenJetCleaner',
  src    = cms.untracked.string('iterativeCone5HiGenJets'),
  deltaR = cms.untracked.double(0.25),
  ptCut  = cms.untracked.double(20),
  createNewCollection = cms.untracked.bool(True),
  fillDummyEntries = cms.untracked.bool(True)
)

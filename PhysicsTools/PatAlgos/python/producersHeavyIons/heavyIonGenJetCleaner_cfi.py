import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.SelectPartons_cff import myPartons
genPartons = myPartons.clone(
    src = cms.InputTag("hiGenParticles")
    )

hiPartons = cms.EDProducer('HiPartonCleaner',
                           src = cms.untracked.InputTag('genPartons'),
                           deltaR = cms.untracked.double(0.25),
                           ptCut  = cms.untracked.double(20),
                           createNewCollection = cms.untracked.bool(True),
                           fillDummyEntries = cms.untracked.bool(True)
                           )

heavyIonCleanedGenJets = cms.EDProducer('HiGenJetCleaner',
  src    = cms.untracked.InputTag('iterativeCone5HiGenJets'),
  deltaR = cms.untracked.double(0.25),
  ptCut  = cms.untracked.double(20),
  createNewCollection = cms.untracked.bool(True),
  fillDummyEntries = cms.untracked.bool(True)
)

heavyIonCleaned = cms.Sequence(genPartons*hiPartons+heavyIonCleanedGenJets)


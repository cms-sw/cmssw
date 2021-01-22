
import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.SelectPartons_cff import myPartons
genPartons = myPartons.clone(
    src = "hiGenParticles"
)

hiPartons = cms.EDProducer('HiPartonCleaner',
                           src = cms.InputTag('genPartons'),
                           deltaR = cms.double(0.25),
                           ptCut  = cms.double(20),
                           createNewCollection = cms.untracked.bool(True),
                           fillDummyEntries = cms.untracked.bool(True)
                           )

heavyIonCleanedGenJets = cms.EDProducer('HiGenJetCleaner',
                                        src    = cms.InputTag('iterativeCone5HiGenJets'),
                                        deltaR = cms.double(0.25),
                                        ptCut  = cms.double(20),
                                        createNewCollection = cms.untracked.bool(True),
                                        fillDummyEntries = cms.untracked.bool(True)
                                        )
heavyIonCleanedTask = cms.Task(genPartons,hiPartons,heavyIonCleanedGenJets)
heavyIonCleaned = cms.Sequence(heavyIonCleanedTask)

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *

ak1HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak1HiGenJets') )

ak2HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak2HiGenJets') )

ak3HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak3HiGenJets') )

ak4HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak4HiGenJets') )

ak5HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak5HiGenJets') )

ak6HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak6HiGenJets') )

ak7HiGenJetsCleaned = heavyIonCleanedGenJets.clone( src = cms.InputTag('ak7HiGenJets') )

hiGenJetsCleaned = cms.Sequence(
ak1HiGenJetsCleaned
+
ak2HiGenJetsCleaned
+
ak3HiGenJetsCleaned
+
ak4HiGenJetsCleaned
+
ak5HiGenJetsCleaned
+
ak6HiGenJetsCleaned
+
ak7HiGenJetsCleaned
)

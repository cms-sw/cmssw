import FWCore.ParameterSet.Config as cms
from RecoHI.HiJetAlgos.HiRecoJets_cff import *
from RecoHI.HiJetAlgos.HiRecoPFJets_cff import *
akVs1PFJets.jetPtMin = 1
akVs1PFJets.src = cms.InputTag("particleFlowTmp")
akVs1CaloJets.jetPtMin = 1
akVs2PFJets.jetPtMin = 1
akVs2PFJets.src = cms.InputTag("particleFlowTmp")
akVs2CaloJets.jetPtMin = 1
akVs3PFJets.jetPtMin = 1
akVs3PFJets.src = cms.InputTag("particleFlowTmp")
akVs3CaloJets.jetPtMin = 1
akVs4PFJets.jetPtMin = 1
akVs4PFJets.src = cms.InputTag("particleFlowTmp")
akVs4CaloJets.jetPtMin = 1
akVs5PFJets.jetPtMin = 1
akVs5PFJets.src = cms.InputTag("particleFlowTmp")
akVs5CaloJets.jetPtMin = 1
akVs6PFJets.jetPtMin = 1
akVs6PFJets.src = cms.InputTag("particleFlowTmp")
akVs6CaloJets.jetPtMin = 1
akPu1PFJets.jetPtMin = 1
akPu1CaloJets.jetPtMin = 1
akPu2PFJets.jetPtMin = 1
akPu2CaloJets.jetPtMin = 1
akPu3PFJets.jetPtMin = 1
akPu3CaloJets.jetPtMin = 1
akPu4PFJets.jetPtMin = 1
akPu4CaloJets.jetPtMin = 1
akPu5PFJets.jetPtMin = 1
akPu5CaloJets.jetPtMin = 1
akPu6PFJets.jetPtMin = 1
akPu6CaloJets.jetPtMin = 1
akVs1PFJets.jetPtMin = 1
akVs1PFJets.src = cms.InputTag("particleFlowTmp")
akVs1CaloJets.jetPtMin = 1
akVs2PFJets.jetPtMin = 1
akVs2PFJets.src = cms.InputTag("particleFlowTmp")
akVs2CaloJets.jetPtMin = 1
akVs3PFJets.jetPtMin = 1
akVs3PFJets.src = cms.InputTag("particleFlowTmp")
akVs3CaloJets.jetPtMin = 1
akVs4PFJets.jetPtMin = 1
akVs4PFJets.src = cms.InputTag("particleFlowTmp")
akVs4CaloJets.jetPtMin = 1
akVs5PFJets.jetPtMin = 1
akVs5PFJets.src = cms.InputTag("particleFlowTmp")
akVs5CaloJets.jetPtMin = 1
akVs6PFJets.jetPtMin = 1
akVs6PFJets.src = cms.InputTag("particleFlowTmp")
akVs6CaloJets.jetPtMin = 1
akPu1PFJets.jetPtMin = 1
akPu1CaloJets.jetPtMin = 1
akPu2PFJets.jetPtMin = 1
akPu2CaloJets.jetPtMin = 1
akPu3PFJets.jetPtMin = 1
akPu3CaloJets.jetPtMin = 1
akPu4PFJets.jetPtMin = 1
akPu4CaloJets.jetPtMin = 1
akPu5PFJets.jetPtMin = 1
akPu5CaloJets.jetPtMin = 1
akPu6PFJets.jetPtMin = 1
akPu6CaloJets.jetPtMin = 1
akVs1PFJets.jetPtMin = 1
akVs1PFJets.src = cms.InputTag("particleFlowTmp")
akVs1CaloJets.jetPtMin = 1
akVs2PFJets.jetPtMin = 1
akVs2PFJets.src = cms.InputTag("particleFlowTmp")
akVs2CaloJets.jetPtMin = 1
akVs3PFJets.jetPtMin = 1
akVs3PFJets.src = cms.InputTag("particleFlowTmp")
akVs3CaloJets.jetPtMin = 1
akVs4PFJets.jetPtMin = 1
akVs4PFJets.src = cms.InputTag("particleFlowTmp")
akVs4CaloJets.jetPtMin = 1
akVs5PFJets.jetPtMin = 1
akVs5PFJets.src = cms.InputTag("particleFlowTmp")
akVs5CaloJets.jetPtMin = 1
akVs6PFJets.jetPtMin = 1
akVs6PFJets.src = cms.InputTag("particleFlowTmp")
akVs6CaloJets.jetPtMin = 1
akPu1PFJets.jetPtMin = 1
akPu1CaloJets.jetPtMin = 1
akPu2PFJets.jetPtMin = 1
akPu2CaloJets.jetPtMin = 1
akPu3PFJets.jetPtMin = 1
akPu3CaloJets.jetPtMin = 1
akPu4PFJets.jetPtMin = 1
akPu4CaloJets.jetPtMin = 1
akPu5PFJets.jetPtMin = 1
akPu5CaloJets.jetPtMin = 1
akPu6PFJets.jetPtMin = 1
akPu6CaloJets.jetPtMin = 1

hiReRecoPFJets = cms.Sequence(
akPu1PFJets
+
akPu2PFJets
+
akPu3PFJets
+
akPu4PFJets
+
akPu5PFJets
+
akPu6PFJets
+
akVs1PFJets
+
akVs2PFJets
+
akVs3PFJets
+
akVs4PFJets
+
akVs5PFJets
+
akVs6PFJets
)

hiReRecoCaloJets = cms.Sequence(
akPu1CaloJets
+
akPu2CaloJets
+
akPu3CaloJets
+
akPu4CaloJets
+
akPu5CaloJets
+
akPu6CaloJets
+
akVs1CaloJets
+
akVs2CaloJets
+
akVs3CaloJets
+
akVs4CaloJets
+
akVs5CaloJets
+
akVs6CaloJets
)

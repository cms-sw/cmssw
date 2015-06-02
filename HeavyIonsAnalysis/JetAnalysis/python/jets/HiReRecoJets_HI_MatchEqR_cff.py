import FWCore.ParameterSet.Config as cms
from RecoHI.HiJetAlgos.HiRecoJets_cff import *
from RecoHI.HiJetAlgos.HiRecoPFJets_cff import *

voronoiBackgroundCaloEqualizeR0p2= voronoiBackgroundCalo.clone(equalizeR=cms.double(0.2))
voronoiBackgroundCaloEqualizeR0p3= voronoiBackgroundCalo.clone(equalizeR=cms.double(0.3))
voronoiBackgroundCaloEqualizeR0p5= voronoiBackgroundCalo.clone(equalizeR=cms.double(0.5))
voronoiBackgroundCaloEqualizeR0p6= voronoiBackgroundCalo.clone(equalizeR=cms.double(0.6))
# voronoiBackgroundCaloEqualizeR0p7= voronoiBackgroundCalo.clone(equalizeR=cms.double(0.7))
# voronoiBackgroundCaloEqualizeR0p8= voronoiBackgroundCalo.clone(equalizeR=cms.double(0.8))
                          
voronoiBackgroundPFEqualizeR0p1= voronoiBackgroundPF.clone(equalizeR=cms.double(0.1))
voronoiBackgroundPFEqualizeR0p2= voronoiBackgroundPF.clone(equalizeR=cms.double(0.2))
voronoiBackgroundPFEqualizeR0p4= voronoiBackgroundPF.clone(equalizeR=cms.double(0.4))
voronoiBackgroundPFEqualizeR0p5= voronoiBackgroundPF.clone(equalizeR=cms.double(0.5))
# voronoiBackgroundPFEqualizeR0p6= voronoiBackgroundPF.clone(equalizeR=cms.double(0.6))
# voronoiBackgroundPFEqualizeR0p7= voronoiBackgroundPF.clone(equalizeR=cms.double(0.7))

akVs1PFJets.jetPtMin = 1
akVs1PFJets.src = cms.InputTag("particleFlowTmp")
akVs1PFJets.bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p1")
akVs1CaloJets.jetPtMin = 1
akVs1CaloJets.bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p2")
akVs2PFJets.jetPtMin = 1
akVs2PFJets.src = cms.InputTag("particleFlowTmp")
akVs2PFJets.bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p2")
akVs2CaloJets.jetPtMin = 1
akVs2CaloJets.bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p3")
akVs3PFJets.jetPtMin = 1
akVs3PFJets.src = cms.InputTag("particleFlowTmp")
akVs3CaloJets.jetPtMin = 1
akVs4PFJets.jetPtMin = 1
akVs4PFJets.src = cms.InputTag("particleFlowTmp")
akVs4PFJets.bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p4")
akVs4CaloJets.jetPtMin = 1
akVs4CaloJets.bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p5")
akVs5PFJets.jetPtMin = 1
akVs5PFJets.src = cms.InputTag("particleFlowTmp")
akVs5PFJets.bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p5")
akVs5CaloJets.jetPtMin = 1
akVs5CaloJets.bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p6")
akVs6PFJets.jetPtMin = 1
akVs6PFJets.src = cms.InputTag("particleFlowTmp")
akVs6PFJets.bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p5") #set to 5 instead of 6 because 6 takes long time 
akVs6CaloJets.jetPtMin = 1
akVs6CaloJets.bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p6") #set to 6 instead of 7 because 7 takes long time
akVs7PFJets.jetPtMin = 1
akVs7PFJets.src = cms.InputTag("particleFlowTmp")
akVs7PFJets.bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p5") #set to 5 instead of 7 because 7 takes long time 
akVs7CaloJets.jetPtMin = 1
akVs7CaloJets.bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p6") #set to 6 instead of 8 because 8 takes long time

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
akPu7PFJets.jetPtMin = 1
akPu7CaloJets.jetPtMin = 1


hiReRecoPFJetsMatchEqR = cms.Sequence(
voronoiBackgroundPFEqualizeR0p1
+
voronoiBackgroundPFEqualizeR0p2
+
voronoiBackgroundPFEqualizeR0p4
+
voronoiBackgroundPFEqualizeR0p5
+
# voronoiBackgroundPFEqualizeR0p6
# +
# voronoiBackgroundPFEqualizeR0p7
# +
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
akPu7PFJets
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
+
akVs7PFJets
)


hiReRecoPFJetsMatchEqRJetRAA = cms.Sequence(
voronoiBackgroundPFEqualizeR0p2
+
voronoiBackgroundPFEqualizeR0p4
+
voronoiBackgroundPFEqualizeR0p5
+
# voronoiBackgroundPFEqualizeR0p6
# +
# voronoiBackgroundPFEqualizeR0p7
# +
akPu2PFJets
+
akPu3PFJets
+
akPu3CaloJets
+
akPu4PFJets
+
akVs2PFJets
+
akVs3CaloJets
+
akVs3PFJets
+
akVs4PFJets
+
akVs5PFJets
)

hiReRecoCaloJetsMatchEqR = cms.Sequence(
voronoiBackgroundCaloEqualizeR0p2
+
voronoiBackgroundCaloEqualizeR0p3
+
voronoiBackgroundCaloEqualizeR0p5
+
voronoiBackgroundCaloEqualizeR0p6
+
# voronoiBackgroundCaloEqualizeR0p7
# +
# voronoiBackgroundCaloEqualizeR0p8
# +
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
akPu7CaloJets
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
+
akVs7CaloJets
)

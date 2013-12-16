import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.SelectPartons_cff import *
from PhysicsTools.JetMCAlgos.AK5CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.AK5CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.AK8CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.GK5CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.GK7CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.KT4CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.KT6CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.SC5CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.SC7CaloJetsMCFlavour_cff import *

ak4Flavour = cms.Sequence(AK5byRef*AK5byValPhys*AK5byValAlgo)
sisCone5Flavour = cms.Sequence(SC5byRef*SC5byValPhys*SC5byValAlgo)
sisCone7Flavour = cms.Sequence(SC7byRef*SC7byValPhys*SC7byValAlgo)
AK5Flavour = cms.Sequence(AK5byRef*AK5byValPhys*AK5byValAlgo)
AK8Flavour = cms.Sequence(AK8byRef*AK8byValPhys*AK8byValAlgo)
GK5Flavour = cms.Sequence(GK5byRef*GK5byValPhys*GK5byValAlgo)
GK7Flavour = cms.Sequence(GK7byRef*GK7byValPhys*GK7byValAlgo)
KT4Flavour = cms.Sequence(KT4byRef*KT4byValPhys*KT4byValAlgo)
KT6Flavour = cms.Sequence(KT6byRef*KT6byValPhys*KT6byValAlgo)

caloJetMCFlavour = cms.Sequence(
	myPartons * (
		ak4Flavour +
		sisCone5Flavour +
		sisCone7Flavour +
		AK5Flavour +
		AK8Flavour +
		GK5Flavour +
		GK7Flavour +
		KT4Flavour +
		KT6Flavour
	)
)

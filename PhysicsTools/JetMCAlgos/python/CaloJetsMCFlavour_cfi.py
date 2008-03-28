import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.SelectPartons_cff import *
from PhysicsTools.JetMCAlgos.IC5CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.KT4CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.KT6CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.SC5CaloJetsMCFlavour_cff import *
from PhysicsTools.JetMCAlgos.SC7CaloJetsMCFlavour_cff import *
iterativeCone5Flavour = cms.Sequence(IC5byRef*IC5byValPhys*IC5byValAlgo)
sisCone5Flavour = cms.Sequence(SC5byRef*SC5byValPhys*SC5byValAlgo)
sisCone7Flavour = cms.Sequence(SC7byRef*SC7byValPhys*SC7byValAlgo)
KT4Flavour = cms.Sequence(KT4byRef*KT4byValPhys*KT4byValAlgo)
KT6Flavour = cms.Sequence(KT6byRef*KT6byValPhys*KT6byValAlgo)
caloJetMCFlavour = cms.Sequence(myPartons*iterativeCone5Flavour+sisCone5Flavour+sisCone7Flavour+KT4Flavour+KT6Flavour)


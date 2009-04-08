# Example cfi file for the CaloJet correction producer. 
# It is used for the HLT confguration database.
import FWCore.ParameterSet.Config as cms
from JetMETCorrections.Modules.JetCorrectionServiceChain_cfi import *

L2L3CorJetIC5Calo = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorIC5Calo')
)

import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak8PFJets_cfi import ak8PFJetsCHSMassDropFiltered, ak8PFJetsCHSFiltered


# Higgs taggers
ca15PFJetsCHSMassDropFiltered = ak8PFJetsCHSMassDropFiltered.clone(
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam = 1.5
    )

ca15PFJetsCHSFiltered = ak8PFJetsCHSFiltered.clone(
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam = 1.5
    )

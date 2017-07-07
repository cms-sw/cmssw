import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets

ak4PFJetsSK = ak4PFJets.clone(
    src = cms.InputTag("softKiller"),
    useExplicitGhosts = cms.bool(True)
    )


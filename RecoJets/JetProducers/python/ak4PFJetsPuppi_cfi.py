import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
import RecoJets.JetProducers.ak4PFJets_cfi

ak4PFJetsPuppi = RecoJets.JetProducers.ak4PFJets_cfi.ak4PFJets.clone(
    src = cms.InputTag("puppi")
    )


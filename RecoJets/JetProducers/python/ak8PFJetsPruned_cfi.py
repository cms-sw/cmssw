import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.ak8PFJets_cfi import ak8PFJets
from RecoJets.JetProducers.SubJetParameters_cfi import SubJetParameters

ak8PFJetsPruned = ak8PFJets.clone(
    SubJetParameters,
    usePruning = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
    )


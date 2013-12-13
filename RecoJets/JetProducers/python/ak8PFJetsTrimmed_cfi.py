import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.ak8PFJets_cfi import ak8PFJets

ak8PFJetsTrimmed = ak8PFJets.clone(
    useTrimming = cms.bool(True),
    rFilt = cms.double(0.2),
    trimPtFracMin = cms.double(0.03),
    useExplicitGhosts = cms.bool(True)
    )


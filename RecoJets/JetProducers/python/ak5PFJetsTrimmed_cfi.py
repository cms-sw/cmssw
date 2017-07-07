import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets

ak5PFJetsTrimmed = ak5PFJets.clone(
    useTrimming = cms.bool(True),
    rFilt = cms.double(0.2),
    trimPtFracMin = cms.double(0.03),
    useExplicitGhosts = cms.bool(True)
    )


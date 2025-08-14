import FWCore.ParameterSet.Config as cms

from Validation.RecoJets.hltJetPostProcessor_cff import *
from HLTriggerOffline.JetMET.Validation.JetMETPostProcessor_cff import *

HLTJetMETPostVal = cms.Sequence(
    recoJetPostProcessorHLT +
    hltJetMETPostProcessor
)
import FWCore.ParameterSet.Config as cms

from Validation.RecoJets.hltJetPostProcessor_cff import hltJetPostProcessor
from Validation.RecoMET.hltMETPostProcessor_cff import hltMETPostProcessor
from HLTriggerOffline.JetMET.Validation.JetMETPostProcessor_cff import hltJetMETPostProcessor

HLTJetMETPostVal = cms.Sequence(
    hltJetPostProcessor +
    hltMETPostProcessor +
    hltJetMETPostProcessor # Jet and MET trigger turn-on curves
)

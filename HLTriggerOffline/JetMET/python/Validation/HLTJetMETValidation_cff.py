import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.JetMET.Validation.SingleJetValidation_cfi import *
from Validation.RecoJets.hltJetValidation_cff import *

##please do NOT include paths here!
HLTJetMETValSeq    = cms.Sequence(
    SingleJetValidation
    + hltJetAnalyzerAk4PFPuppi
    + hltJetAnalyzerAk4PFCluster
    + hltJetAnalyzerAk4PF
    + hltJetAnalyzerAk4PFCHS
)
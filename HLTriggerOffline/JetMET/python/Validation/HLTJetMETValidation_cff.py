import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.JetMET.Validation.SingleJetValidation_cfi import *
from Validation.RecoJets.hltJetValidation_cff import *
from Validation.RecoMET.hltMETValidation_cff import *

##please do NOT include paths here!
HLTJetMETValSeq = cms.Sequence(
    SingleJetValidation
    + hltJetAnalyzerAK4PFPuppi
    + hltJetAnalyzerAK4PF
    + hltJetAnalyzerAK4PFCHS
    + hltMetAnalyzerPF
    + hltMetAnalyzerPFCalo
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(HLTJetMETValSeq, 
    cms.Sequence(
        HLTJetMETValSeq.copy()
        + hltMetAnalyzerPFPuppi
        + hltMetTypeOneAnalyzerPFPuppi
    )
)

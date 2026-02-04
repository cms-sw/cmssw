import FWCore.ParameterSet.Config as cms
from functools import reduce

from HLTriggerOffline.JetMET.Validation.SingleJetValidation_cfi import *
from Validation.RecoJets.hltJetValidation_cff import *
from Validation.RecoMET.hltMETValidation_cff import *

from RecoMET.Configuration.GenMHT_cff import genMHT

def _sumModules(alist):
    return reduce(lambda x, y: x + y, alist)

metmht_common_analyzers = [hltMetAnalyzerPF, hltMetAnalyzerPFCalo, hltMetTypeOneAnalyzerPF] 
# metmht_run3_analyzers = metmht_common_analyzers + [hltMetAnalyzerPFNoMu, hltMhtAnalyzer, hltMhtAnalyzerPFTightID]
metmht_run3_analyzers = metmht_common_analyzers + [hltMetAnalyzerPFNoMu]
metmht_ph2_analyzers =  metmht_common_analyzers + [hltMetAnalyzerPFPuppi, hltMhtAnalyzerPFPuppi]

##please do NOT include paths here!
HLTJetMETValSeq = cms.Sequence(
    SingleJetValidation
    + hltJetAnalyzerAK4PFPuppi
    + hltJetAnalyzerAK4PF
    + hltJetAnalyzerAK4PFCHS
    + genMHT
    + _sumModules(metmht_run3_analyzers)
)

_phase2_HLTJetMETValSeq = HLTJetMETValSeq.copyAndExclude(metmht_run3_analyzers)
_phase2_HLTJetMETValSeq += cms.Sequence(_sumModules(metmht_ph2_analyzers))

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(HLTJetMETValSeq, _phase2_HLTJetMETValSeq)

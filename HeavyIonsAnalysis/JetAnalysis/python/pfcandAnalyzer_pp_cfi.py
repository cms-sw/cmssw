import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_cfi import *

pfcandAnalyzer.pfCandidateLabel = cms.InputTag("particleFlow")


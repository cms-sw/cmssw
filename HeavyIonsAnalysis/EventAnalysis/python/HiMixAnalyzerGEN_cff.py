import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.EventAnalysis.HiMixAnalyzer_cfi import *

mixAnalyzer.doRECO = True
mixAnalyzer.jetSrc = cms.untracked.InputTag('akPu4CaloJets')




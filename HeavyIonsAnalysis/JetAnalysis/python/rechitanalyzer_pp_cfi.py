import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_cfi import *

rechitanalyzer.vtxSrc = cms.untracked.InputTag("offlinePrimaryVerticesWithBS")
rechitanalyzer.JetSrc = cms.untracked.InputTag("ak3CaloJets")

pfTowers.vtxSrc = cms.untracked.InputTag("offlinePrimaryVerticesWithBS")
pfTowers.JetSrc = cms.untracked.InputTag("ak3CaloJets")

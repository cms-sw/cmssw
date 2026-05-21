import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akCs4PFJetAnalyzer = inclusiveJetAnalyzer.clone(
    jetTag = cms.InputTag("slimmedJets"),
    rParam = 0.4,
    fillGenJets = False,
    isMC = False,
    jetName = cms.untracked.string("akCs4PF"),
    hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
    )

akFlowPuCs4PFJetAnalyzer = akCs4PFJetAnalyzer.clone()

import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

inclusiveJetAnalyzer.doHiJetID = False
inclusiveJetAnalyzer.isMC = False
inclusiveJetAnalyzer.usePAT = cms.untracked.bool(False)
inclusiveJetAnalyzer.useVtx = cms.untracked.bool(False)
inclusiveJetAnalyzer.useJEC = cms.untracked.bool(False)

ak3GenJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak3HiGenJets"))
ak4GenJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak4HiGenJets"))
ak5GenJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak5HiGenJets"))


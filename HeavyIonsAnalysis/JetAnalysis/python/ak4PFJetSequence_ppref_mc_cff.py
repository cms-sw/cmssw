import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak4PFJetAnalyzer = inclusiveJetAnalyzer.clone(
    jetTag = cms.InputTag("slimmedJets"),
    genjetTag = 'slimmedGenJets',
    rParam = 0.4,
    fillGenJets = True,
    isMC = True,
    doSubEvent = False,
    useHepMC = cms.untracked.bool(False),
    genParticles = cms.untracked.InputTag("prunedGenParticles"),
    eventInfoTag = cms.InputTag("generator"),
    jetName = cms.untracked.string("ak4PF"),
    genPtMin = cms.untracked.double(5),
    hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
    genDroppedBranches = cms.InputTag("ak4GenJets","droppedBranches"),
    jetFlavourInfos = cms.InputTag("ak4PFFlavourInfos")
    )

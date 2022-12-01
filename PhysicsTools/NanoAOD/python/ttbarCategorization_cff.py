import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.globalVariablesTableProducer_cfi import globalVariablesTableProducer

##################### User floats producers, selectors ##########################

from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenBHadron
matchGenBHadron = matchGenBHadron.clone(
    genParticles = cms.InputTag("prunedGenParticles"),
    jetFlavourInfos = cms.InputTag("slimmedGenJetsFlavourInfos"),
)

## Plugin for analysing C hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenCHadron
matchGenCHadron = matchGenCHadron.clone(
    genParticles = cms.InputTag("prunedGenParticles"),
    jetFlavourInfos = cms.InputTag("slimmedGenJetsFlavourInfos"),
)

## Producer for ttbar categorisation ID
from TopQuarkAnalysis.TopTools.categorizeGenTtbar_cfi import categorizeGenTtbar
categorizeGenTtbar = categorizeGenTtbar.clone(
    genJetPtMin = 20.,
    genJetAbsEtaMax = 2.4,
    genJets = cms.InputTag("slimmedGenJets"),
)


##################### Tables for final output and docs ##########################
ttbarCategoryTable = globalVariablesTableProducer.clone(
    variables = cms.PSet(
        genTtbarId = ExtVar( cms.InputTag("categorizeGenTtbar:genTtbarId"), "int", doc = "ttbar categorization")
    )
)

ttbarCategoryTableTask = cms.Task(ttbarCategoryTable)
ttbarCatMCProducersTask = cms.Task(matchGenBHadron,matchGenCHadron,categorizeGenTtbar)

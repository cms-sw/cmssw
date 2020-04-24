from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_92X_cff import run2_nanoAOD_92X


import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *


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
from TopQuarkAnalysis.TopTools.GenTtbarCategorizer_cfi import categorizeGenTtbar
categorizeGenTtbar = categorizeGenTtbar.clone(
    genJetPtMin = 20.,
    genJetAbsEtaMax = 2.4,
    genJets = cms.InputTag("slimmedGenJets"),
)


### Era dependent customization
run2_miniAOD_80XLegacy.toModify( matchGenBHadron, jetFlavourInfos = cms.InputTag("genJetFlavourAssociation"),)
run2_nanoAOD_92X.toModify( matchGenBHadron, jetFlavourInfos = cms.InputTag("genJetFlavourAssociation"),)

run2_miniAOD_80XLegacy.toModify( matchGenCHadron, jetFlavourInfos = cms.InputTag("genJetFlavourAssociation"),)
run2_nanoAOD_92X.toModify( matchGenCHadron, jetFlavourInfos = cms.InputTag("genJetFlavourAssociation"),)


##################### Tables for final output and docs ##########################
ttbarCategoryTable = cms.EDProducer("GlobalVariablesTableProducer",
                                    variables = cms.PSet(
                                        genTtbarId = ExtVar( cms.InputTag("categorizeGenTtbar:genTtbarId"), "int", doc = "ttbar categorization")
                                    )
)

ttbarCatMCProducers = cms.Sequence(matchGenBHadron + matchGenCHadron + categorizeGenTtbar)

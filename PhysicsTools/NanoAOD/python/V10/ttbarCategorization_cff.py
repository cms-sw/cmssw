import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *


##################### User floats producers, selectors ##########################

matchGenBHadron = cms.EDProducer("GenHFHadronMatcher",
    flavour = cms.int32(5),
    genParticles = cms.InputTag("prunedGenParticles"),
    jetFlavourInfos = cms.InputTag("slimmedGenJetsFlavourInfos"),
    noBBbarResonances = cms.bool(False),
    onlyJetClusteredHadrons = cms.bool(True)
)

## Plugin for analysing C hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
matchGenCHadron = cms.EDProducer("GenHFHadronMatcher",
    flavour = cms.int32(4),
    genParticles = cms.InputTag("prunedGenParticles"),
    jetFlavourInfos = cms.InputTag("slimmedGenJetsFlavourInfos"),
    noBBbarResonances = cms.bool(False),
    onlyJetClusteredHadrons = cms.bool(True)
)

## Producer for ttbar categorisation ID
categorizeGenTtbar = cms.EDProducer("GenTtbarCategorizer",
    genBHadFlavour = cms.InputTag("matchGenBHadron","genBHadFlavour"),
    genBHadFromTopWeakDecay = cms.InputTag("matchGenBHadron","genBHadFromTopWeakDecay"),
    genBHadIndex = cms.InputTag("matchGenBHadron","genBHadIndex"),
    genBHadJetIndex = cms.InputTag("matchGenBHadron","genBHadJetIndex"),
    genBHadLeptonHadronIndex = cms.InputTag("matchGenBHadron","genBHadLeptonHadronIndex"),
    genBHadLeptonViaTau = cms.InputTag("matchGenBHadron","genBHadLeptonViaTau"),
    genBHadPlusMothers = cms.InputTag("matchGenBHadron","genBHadPlusMothers"),
    genBHadPlusMothersIndices = cms.InputTag("matchGenBHadron","genBHadPlusMothersIndices"),
    genCHadBHadronId = cms.InputTag("matchGenCHadron","genCHadBHadronId"),
    genCHadFlavour = cms.InputTag("matchGenCHadron","genCHadFlavour"),
    genCHadFromTopWeakDecay = cms.InputTag("matchGenCHadron","genCHadFromTopWeakDecay"),
    genCHadJetIndex = cms.InputTag("matchGenCHadron","genCHadJetIndex"),
    genJetAbsEtaMax = cms.double(2.4),
    genJetPtMin = cms.double(20),
    genJets = cms.InputTag("slimmedGenJets"),
    mightGet = cms.optional.untracked.vstring
)


##################### Tables for final output and docs ##########################
ttbarCategoryTable = cms.EDProducer("GlobalVariablesTableProducer",
                                    variables = cms.PSet(
                                        genTtbarId = ExtVar( cms.InputTag("categorizeGenTtbar:genTtbarId"), "int", doc = "ttbar categorization")
                                    )
)

ttbarCategoryTableTask = cms.Task(ttbarCategoryTable)
ttbarCatMCProducersTask = cms.Task(matchGenBHadron,matchGenCHadron,categorizeGenTtbar)

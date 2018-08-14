from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask
import sys
import os


## Define the process
process = cms.Process("Filter")
patAlgosToolsTask = getPatAlgosToolsTask(process)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
## Set up command line options
options = VarParsing ('analysis')
options.register('runOnGenOrAODsim', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "GEN SIM")
options.register( "skipEvents", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Number of events to skip" )
options.parseArguments()


## Configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 100

## Define input
if not options.inputFiles:
    if options.runOnGenOrAODsim:
        options.inputFiles=['/store/mc/RunIISummer15GS/TTToSemiLeptonic_TuneCUETP8M1_alphaS01273_13TeV-powheg-scaledown-pythia8/GEN-SIM/MCRUN2_71_V1-v2/40000/DE7952A2-6E2F-E611-A803-001E673D1B21.root']
    else:
        options.inputFiles=['/store/mc/RunIIFall17MiniAOD/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/MINIAODSIM/94X_mc2017_realistic_v10-v1/50000/DC5D3109-F2E1-E711-A26E-A0369FC5FC9C.root']

## Define maximum number of events to loop over
if options.maxEvents is -1: # maxEvents is set in VarParsing class by default to -1
    options.maxEvents = 1000 # reset for testing

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(int(options.maxEvents)))
process.source = cms.Source(  "PoolSource",
                              fileNames = cms.untracked.vstring(options.inputFiles),
                              skipEvents=cms.untracked.uint32(int(options.skipEvents)),
)

## Set input particle collections to be used by the tools
genParticleCollection = ''
genJetInputParticleCollection = ''
genJetCollection = 'ak4GenJetsCustom'

if options.runOnGenOrAODsim:
    genParticleCollection = 'genParticles'
    genJetInputParticleCollection = genParticleCollection
else:
    genParticleCollection = 'prunedGenParticles'
    genJetInputParticleCollection = 'packedGenParticles'

## Supplies PDG ID to real name resolution of MC particles
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
## producing a subset of particles to be used for jet clustering

from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJetsNoNu
process.genParticlesForJetsNoNu = genParticlesForJetsNoNu.clone(
	src = genJetInputParticleCollection
)
patAlgosToolsTask.add(process.genParticlesForJetsNoNu)

## Produce own jets (re-clustering in miniAOD needed at present to avoid crash)
from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
process.ak4GenJetsCustom = ak4GenJets.clone(
    src = 'genParticlesForJetsNoNu',
    rParam = cms.double(0.4),
    jetAlgorithm = cms.string("AntiKt")
)
patAlgosToolsTask.add(process.ak4GenJetsCustom)

## Ghost particle collection used for Hadron-Jet association
# MUST use proper input particle collection
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
process.selectedHadronsAndPartons = selectedHadronsAndPartons.clone(
    particles = genParticleCollection,
)
patAlgosToolsTask.add(process.selectedHadronsAndPartons)
#Input particle collection for matching to gen jets (partons + leptons)
# MUST use use proper input jet collection: the jets to which hadrons should be associated
# rParam and jetAlgorithm MUST match those used for jets to be associated with hadrons
# More details on the tool: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools#New_jet_flavour_definition
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
process.genJetFlavourInfos = ak4JetFlavourInfos.clone(
    jets = genJetCollection,
)
patAlgosToolsTask.add(process.genJetFlavourInfos)

# Plugin for analysing B hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenBHadron
process.matchGenBHadron = matchGenBHadron.clone(
    genParticles = genParticleCollection,
    jetFlavourInfos = "genJetFlavourInfos",
    onlyJetClusteredHadrons = cms.bool(False)
)

# Plugin for analysing C hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
#from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenCHadron
#process.matchGenCHadron = matchGenCHadron.clone(
#    genParticles = genParticleCollection,
#    jetFlavourInfos = "genJetFlavourInfos"
#)

process.load("PhysicsTools/JetMCAlgos/ttHFGenFilter_cfi")
from PhysicsTools.JetMCAlgos.ttHFGenFilter_cfi import ttHFGenFilter
process.ttHFGenFilter = ttHFGenFilter.clone(
    genParticles = genParticleCollection,
    taggingMode  = cms.bool(True),
)
print("If taggingMode is set to true, the filter will write a branch into the tree instead of filtering the events")
print("taggingMode is set to ", process.ttHFGenFilter.taggingMode)



## configuring the testing analyzer that produces output tree
#process.matchGenHFHadrons = cms.EDAnalyzer("matchGenHFHadrons",
#    # phase space of jets to be stored
#    genJetPtMin = cms.double(15),
#    genJetAbsEtaMax = cms.double(2.4),
#    # input tags holding information about matching
#    genJets = cms.InputTag(genJetCollection),
#    genBHadJetIndex = cms.InputTag("matchGenBHadron", "genBHadJetIndex"),
#    genBHadFlavour = cms.InputTag("matchGenBHadron", "genBHadFlavour"),
#    genBHadFromTopWeakDecay = cms.InputTag("matchGenBHadron", "genBHadFromTopWeakDecay"),
#    genBHadPlusMothers = cms.InputTag("matchGenBHadron", "genBHadPlusMothers"),
#    genBHadPlusMothersIndices = cms.InputTag("matchGenBHadron", "genBHadPlusMothersIndices"),
#    genBHadIndex = cms.InputTag("matchGenBHadron", "genBHadIndex"),
#    genBHadLeptonHadronIndex = cms.InputTag("matchGenBHadron", "genBHadLeptonHadronIndex"),
#    genBHadLeptonViaTau = cms.InputTag("matchGenBHadron", "genBHadLeptonViaTau"),
#    genCHadJetIndex = cms.InputTag("matchGenCHadron", "genCHadJetIndex"),
#    genCHadFlavour = cms.InputTag("matchGenCHadron", "genCHadFlavour"),
#    genCHadFromTopWeakDecay = cms.InputTag("matchGenCHadron", "genCHadFromTopWeakDecay"),
#    genCHadBHadronId = cms.InputTag("matchGenCHadron", "genCHadBHadronId"),
#    genCHadPlusMothers = cms.InputTag("matchGenCHadron", "genCHadPlusMothers"),
#    genCHadPlusMothersIndices = cms.InputTag("matchGenCHadron", "genCHadPlusMothersIndices"),
#    genCHadIndex = cms.InputTag("matchGenCHadron", "genCHadIndex"),
#    genCHadLeptonHadronIndex = cms.InputTag("matchGenCHadron", "genCHadLeptonHadronIndex"),
#    genCHadLeptonViaTau = cms.InputTag("matchGenCHadron", "genCHadLeptonViaTau"),
#)

## Configure test filter
#process.ttHFGenFilter = cms.EDFilter("ttHFGenFilter",
#    genBHadFlavour = cms.InputTag("matchGenBHadron", "genBHadFlavour"),
#    genBHadFromTopWeakDecay = cms.InputTag("matchGenBHadron", "genBHadFromTopWeakDecay"),
#    genBHadPlusMothers = cms.InputTag("matchGenBHadron", "genBHadPlusMothers"),
#    genBHadPlusMothersIndices = cms.InputTag("matchGenBHadron", "genBHadPlusMothersIndices"),
#    genBHadIndex = cms.InputTag("matchGenBHadron", "genBHadIndex"),
#    filter = cms.bool(True)
#)
process.USER = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1')
    ),
    fileName = cms.untracked.string('Filtered_Events.root')
)
## Output root file
#process.TFileService = cms.Service("TFileService",
#    fileName = cms.string("genHFHadronMatcherOutput.root")
#)

## Path
process.p1 = cms.Path(
    process.matchGenBHadron*
    process.ttHFGenFilter
)

process.endpath = cms.EndPath(process.USER, patAlgosToolsTask)

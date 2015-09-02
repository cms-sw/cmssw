import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import sys

process = cms.Process("Analyzer")

## defining command line options
options = VarParsing.VarParsing ('standard')
options.register('runOnAOD', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "run on AOD")

## processing provided options
if( hasattr(sys, "argv") ):
    for args in sys.argv :
        arg = args.split(',')
        for val in arg:
             val = val.split('=')
             if(len(val)==2):
                 setattr(options,val[0], val[1])

## enabling unscheduled mode for modules
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    allowUnscheduled = cms.untracked.bool(True),
)

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 100

## define input
if options.runOnAOD:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(    
          ## add your favourite AOD files here (1000+ events in each file defined below)
	  '/store/mc/RunIISpring15DR74/ttHTobb_M125_13TeV_powheg_pythia8/AODSIM/Asympt25ns_MCRUN2_74_V9-v1/00000/02203C96-2108-E511-B5C1-00266CFCC214.root',
	  '/store/mc/RunIISpring15DR74/ttHTobb_M125_13TeV_powheg_pythia8/AODSIM/Asympt25ns_MCRUN2_74_V9-v1/00000/02332898-9808-E511-81E8-3417EBE34BFD.root',
	  '/store/mc/RunIISpring15DR74/ttHTobb_M125_13TeV_powheg_pythia8/AODSIM/Asympt25ns_MCRUN2_74_V9-v1/00000/060177B4-2E08-E511-83AD-3417EBE644DA.root',
	  ## other AOD samples
#	  '/store/mc/RunIISpring15DR74/TT_TuneCUETP8M1_13TeV-powheg-pythia8/AODSIM/Asympt50ns_MCRUN2_74_V9A-v4/10000/00199A75-540F-E511-8277-0002C92DB464.root',
#	  '/store/mc/RunIISpring15DR74/TT_TuneCUETP8M1_13TeV-powheg-pythia8/AODSIM/Asympt50ns_MCRUN2_74_V9A-v4/10000/001B27B8-550F-E511-85CF-0025905A609A.root',
#	  '/store/mc/RunIISpring15DR74/TT_TuneCUETP8M1_13TeV-powheg-pythia8/AODSIM/Asympt50ns_MCRUN2_74_V9A-v4/10000/007E116A-510F-E511-8D40-F04DA275C007.root',
        ),
	skipEvents = cms.untracked.uint32(0)
    )
else:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(    
	  # add your favourite miniAOD files here (1000+ events in each file defined below)
	  '/store/mc/RunIISpring15DR74/ttHTobb_M125_13TeV_powheg_pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v1/00000/141B9915-1F08-E511-B9FF-001E675A6AB3.root',
#	  '/store/mc/RunIISpring15DR74/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/Asympt50ns_MCRUN2_74_V9A-v4/10000/00D2A247-2910-E511-9F3D-0CC47A4DEDD2.root',
	),
	skipEvents = cms.untracked.uint32(0)
    )

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Setting input particle/jet collections to be used by the tools
genParticleCollection = ''
genJetCollection = 'ak4GenJetsCustom'

if options.runOnAOD:
    genParticleCollection = 'genParticles'
    ## producing a subset of genParticles to be used for jet reclustering
    from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJetsNoNu
    process.genParticlesForJetsCustom = genParticlesForJetsNoNu.clone(
        src = genParticleCollection
    )
    # Producing own jets for testing purposes
    from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
    process.ak4GenJetsCustom = ak4GenJets.clone(
        src = 'genParticlesForJetsCustom',
        rParam = cms.double(0.4),
        jetAlgorithm = cms.string("AntiKt")
    )
else:
    genParticleCollection = 'prunedGenParticles'
    genJetCollection = 'slimmedGenJets'

# Supplies PDG ID to real name resolution of MC particles
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Ghost particle collection used for Hadron-Jet association 
# MUST use proper input particle collection
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
process.selectedHadronsAndPartons = selectedHadronsAndPartons.clone(
    particles = genParticleCollection
)

# Input particle collection for matching to gen jets (partons + leptons) 
# MUST use use proper input jet collection: the jets to which hadrons should be associated
# rParam and jetAlgorithm MUST match those used for jets to be associated with hadrons
# More details on the tool: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools#New_jet_flavour_definition
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
process.genJetFlavourInfos = ak4JetFlavourInfos.clone(
    jets = genJetCollection,
)

# Plugin for analysing B hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenBHadron
process.matchGenBHadron = matchGenBHadron.clone(
    genParticles = genParticleCollection,
    jetFlavourInfos = "genJetFlavourInfos"
)

# Plugin for analysing C hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cff import matchGenCHadron
process.matchGenCHadron = matchGenCHadron.clone(
    genParticles = genParticleCollection,
    jetFlavourInfos = "genJetFlavourInfos"
)


## configuring the testing analyzer that produces output tree
process.matchGenHFHadrons = cms.EDAnalyzer("matchGenHFHadrons",
    # phase space of jets to be stored
    genJetPtMin = cms.double(20),
    genJetAbsEtaMax = cms.double(2.4),
    # input tags holding information about matching
    genJets = cms.InputTag(genJetCollection),
    genBHadJetIndex = cms.InputTag("matchGenBHadron", "genBHadJetIndex"),
    genBHadFlavour = cms.InputTag("matchGenBHadron", "genBHadFlavour"),
    genBHadFromTopWeakDecay = cms.InputTag("matchGenBHadron", "genBHadFromTopWeakDecay"),
    genBHadPlusMothers = cms.InputTag("matchGenBHadron", "genBHadPlusMothers"),
    genBHadPlusMothersIndices = cms.InputTag("matchGenBHadron", "genBHadPlusMothersIndices"),
    genBHadIndex = cms.InputTag("matchGenBHadron", "genBHadIndex"),
    genBHadLeptonHadronIndex = cms.InputTag("matchGenBHadron", "genBHadLeptonHadronIndex"),
    genBHadLeptonViaTau = cms.InputTag("matchGenBHadron", "genBHadLeptonViaTau"),
    genCHadJetIndex = cms.InputTag("matchGenCHadron", "genCHadJetIndex"),
    genCHadFlavour = cms.InputTag("matchGenCHadron", "genCHadFlavour"),
    genCHadFromTopWeakDecay = cms.InputTag("matchGenCHadron", "genCHadFromTopWeakDecay"),
    genCHadBHadronId = cms.InputTag("matchGenCHadron", "genCHadBHadronId"),
    genCHadPlusMothers = cms.InputTag("matchGenCHadron", "genCHadPlusMothers"),
    genCHadPlusMothersIndices = cms.InputTag("matchGenCHadron", "genCHadPlusMothersIndices"),
    genCHadIndex = cms.InputTag("matchGenCHadron", "genCHadIndex"),
    genCHadLeptonHadronIndex = cms.InputTag("matchGenCHadron", "genCHadLeptonHadronIndex"),
    genCHadLeptonViaTau = cms.InputTag("matchGenCHadron", "genCHadLeptonViaTau"),
)

## setting up output root file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("matchGenHFHadrons_trees.root")
)



## defining only the final modules to run: dependencies will be run automatically [allowUnscheduled = True]
process.p1 = cms.Path(
    process.matchGenHFHadrons
)

## module to store raw output from the processed modules into the ROOT file
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('matchGenHFHadrons_out.root'),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_matchGen*_*_*')
    )
process.outpath = cms.EndPath(process.out)

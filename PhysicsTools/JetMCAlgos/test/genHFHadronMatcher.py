import FWCore.ParameterSet.Config as cms

process = cms.Process("Analyzer")

## setting the summary of the process
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 100

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(    
      ## add your favourite AOD files here (1000+ events in each file defined below)
      # Madspin ttbar+jets
      '/store/mc/Summer12_DR53X/TTJets_MSDecays_central_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S10_START53_V19-v1/00000/003CFA73-E945-E311-8917-00266CFAE318.root',
      # Pythia6 ttH(bb)
# '/store/mc/Summer12_DR53X/TTH_HToBB_M-125_8TeV-pythia6/AODSIM/PU_S10_START53_V7A-v1/0000/FCECDD8C-8FFC-E111-AC52-00215E2283D6.root',
    ),
    skipEvents = cms.untracked.uint32(0)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Setting input particle collections to be used by the tools
genParticleCollection = 'genParticles'
genJetInputParticleCollection = 'genParticlesForJets'
genJetCollection = 'ak5GenJetsCustom'

# Supplies PDG ID to real name resolution of MC particles
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Input particles for gen jets (stable gen particles, excluding: electrons, muons and neutrinos from hard interaction)
from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJets
process.genParticlesForJets = genParticlesForJets.clone()

# Producing own jets for testing purposes
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
process.ak5GenJetsCustom = ak5GenJets.clone(
    src = 'genParticlesForJets',
    rParam = cms.double(0.5),
    jetAlgorithm = cms.string("AntiKt")
)

## loading sequences for b/c hadron matching
process.load("PhysicsTools.JetMCAlgos.sequences.GenHFHadronMatching_cff")

# Input particle collection for matching to gen jets (partons + leptons) 
# MUST use use proper input jet collection: the jets to which hadrons should be associated
# rParam and jetAlgorithm MUST match those used for jets to be associated with hadrons
# More details on the tool: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools#New_jet_flavour_definition
process.genJetFlavourPlusLeptonInfos.jets = genJetCollection
process.genJetFlavourPlusLeptonInfos.rParam = cms.double(0.5)
process.genJetFlavourPlusLeptonInfos.jetAlgorithm = cms.string("AntiKt")


## configuring the testing analyzer that produces output tree
process.genHFHadronMatcher = cms.EDAnalyzer("genHFHadronMatcher",
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
)

## setting up output root file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string("genHFHadronMatcher_trees.root")
)


## defining only the final modules to run: dependencies will be run automatically [allowUnscheduled = True]
process.p1 = cms.Path(
    process.genParticlesForJets *
    process.ak5GenJetsCustom *
    process.matchGenBCHadronSequence *
    process.genHFHadronMatcher
)

## module to store raw output from the processed modules into the ROOT file
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('genHFHadronMatcher_out.root'),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_matchGen*_*_*')
    )
process.outpath = cms.EndPath(process.out)

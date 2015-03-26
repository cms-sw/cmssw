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
    wantSummary = cms.untracked.bool(True),
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
#	  '/store/relval/CMSSW_7_2_0_pre7/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V12-v1/00000/1267B7ED-2F4E-E411-A0B9-0025905964A6.root',
	  ## other AOD samples
	  '/store/mc/Spring14dr/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/AODSIM/PU_S14_POSTLS170_V6-v1/00000/00120F7A-84F5-E311-9FBE-002618943910.root',
#	  '/store/mc/Spring14dr/TT_Tune4C_13TeV-pythia8-tauola/AODSIM/Flat20to50_POSTLS170_V5-v1/00000/023E1847-ADDC-E311-91A2-003048FFD754.root',
#	  '/store/mc/Spring14dr/TTbarH_HToBB_M-125_13TeV_pythia6/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/1CAB7E58-0BD0-E311-B688-00266CFFBC3C.root',
#	  '/store/mc/Spring14dr/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/0E3D08A9-C610-E411-A862-0025B3E0657E.root',
        ),
	skipEvents = cms.untracked.uint32(0)
    )
else:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(    
	  # add your favourite miniAOD files here (1000+ events in each file defined below)
	  '/store/cmst3/user/gpetrucc/miniAOD/v1/TTbarH_HToBB_M-125_13TeV_pythia6_PU_S14_PAT.root',
	),
	skipEvents = cms.untracked.uint32(0)
    )

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Setting input particle collections to be used by the tools
genParticleCollection = ''
genJetInputParticleCollection = ''
genJetCollection = 'ak4GenJetsCustom'
if options.runOnAOD:
    genParticleCollection = 'genParticles'
    genJetInputParticleCollection = 'genParticlesForJets'
    ## producing a subset of genParticles to be used for jet clustering in AOD
    from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJets
    process.genParticlesForJets = genParticlesForJets.clone()
else:
    genParticleCollection = 'prunedGenParticles'
    genJetInputParticleCollection = 'packedGenParticles'

# Supplies PDG ID to real name resolution of MC particles
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Producing own jets for testing purposes
from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
process.ak4GenJetsCustom = ak4GenJets.clone(
    src = genJetInputParticleCollection,
    rParam = cms.double(0.4),
    jetAlgorithm = cms.string("AntiKt")
)

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
from PhysicsTools.JetMCAlgos.sequences.GenHFHadronMatching_cff import genJetFlavourPlusLeptonInfos
process.genJetFlavourPlusLeptonInfos = genJetFlavourPlusLeptonInfos.clone(
    jets = genJetCollection,
    rParam = cms.double(0.4),
    jetAlgorithm = cms.string("AntiKt")
)


# Plugin for analysing B hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.sequences.GenHFHadronMatching_cff import matchGenBHadron
process.matchGenBHadron = matchGenBHadron.clone(
    genParticles = genParticleCollection
)

# Plugin for analysing C hadrons
# MUST use the same particle collection as in selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.sequences.GenHFHadronMatching_cff import matchGenCHadron
process.matchGenCHadron = matchGenCHadron.clone(
    genParticles = genParticleCollection
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

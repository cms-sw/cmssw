import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # /TTJets_MassiveBinDECAY_TuneZ2star_8TeV-madgraph-tauola/Summer12_DR53X-PU_S10_START53_V7C-v1/AODSIM
        '/store/mc/Summer12_DR53X/TTJets_MassiveBinDECAY_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S10_START53_V7C-v1/00000/008BD264-1526-E211-897A-00266CFFA7BC.root'
    )
)

process.printList = cms.EDAnalyzer("ParticleListDrawer",
    src = cms.InputTag("genParticles"),
    maxEventsToPrint = cms.untracked.int32(1)
)
#-------------------------------------------------------------------------
# AK5 jets
#-------------------------------------------------------------------------
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
process.selectedHadronsAndPartons = selectedHadronsAndPartons.clone()

from PhysicsTools.JetMCAlgos.AK5PFJetsMCFlavourInfos_cfi import ak5JetFlavourInfos
process.jetFlavourInfosAK5PFJets = ak5JetFlavourInfos.clone()

process.printEventAK5PFJets = cms.EDAnalyzer("printJetFlavourInfo",
    jetFlavourInfos = cms.InputTag("jetFlavourInfosAK5PFJets")
)
#-------------------------------------------------------------------------
# AK8 fat jets and pruned subjets
#-------------------------------------------------------------------------
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
process.ak8PFJets = ak5PFJets.clone(
    rParam        = cms.double(0.8),
    src           = cms.InputTag("particleFlow"),
)

from RecoJets.JetProducers.ak5PFJetsPruned_cfi import ak5PFJetsPruned
process.ak8PFJetsPruned = ak5PFJetsPruned.clone(
    rParam              = cms.double(0.8),
    src                 = cms.InputTag("particleFlow"),
    writeCompound       = cms.bool(True),
    jetCollInstanceName = cms.string("SubJets")
)

process.jetFlavourInfosAK8PFJets = cms.EDProducer("JetFlavourClustering",
    jets                     = cms.InputTag("ak8PFJets"),
    groomedJets              = cms.InputTag("ak8PFJetsPruned"),
    subjets                  = cms.InputTag("ak8PFJetsPruned", "SubJets"),
    bHadrons                 = cms.InputTag("selectedHadronsAndPartons","bHadrons"),
    cHadrons                 = cms.InputTag("selectedHadronsAndPartons","cHadrons"),
    partons                  = cms.InputTag("selectedHadronsAndPartons","algorithmicPartons"),
    leptons                  = cms.InputTag("selectedHadronsAndPartons","leptons"),
    jetAlgorithm             = cms.string("AntiKt"),
    rParam                   = cms.double(0.8),
    ghostRescaling           = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(False)
)

process.printEventAK8PFJets = cms.EDAnalyzer("printJetFlavourInfo",
    jetFlavourInfos    = cms.InputTag("jetFlavourInfosAK8PFJets"),
    subjetFlavourInfos = cms.InputTag("jetFlavourInfosAK8PFJets","SubJets"),
    groomedJets        = cms.InputTag("ak8PFJetsPruned"),
)
#-------------------------------------------------------------------------
# CA15 fat jets and HEPTopTagger fat jets and subjets
#-------------------------------------------------------------------------
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.CATopJetParameters_cfi import *
from RecoJets.JetProducers.PFJetParameters_cfi import *

from RecoJets.JetProducers.ca4PFJets_cfi import ca4PFJets
process.ca15PFJets = ca4PFJets.clone(
    rParam   = cms.double(1.5),
    src      = cms.InputTag("particleFlow"),
    jetPtMin = cms.double(100.0)
)

process.caHEPTopTagJets = cms.EDProducer(
    "CATopJetProducer",
    PFJetParameters.clone( src = cms.InputTag("particleFlow"),
                           doAreaFastjet = cms.bool(False),
                           doRhoFastjet = cms.bool(False),
                           jetPtMin = cms.double(100.0)
                           ),
    AnomalousCellParameters,
    CATopJetParameters.clone( tagAlgo = cms.int32(2) ),
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam = cms.double(1.5),
    muCut = cms.double(0.8),
    maxSubjetMass = cms.double(30.0),
    useSubjetMass = cms.bool(False),
    writeCompound = cms.bool(True)
)

process.jetFlavourInfosCA15PFJets = cms.EDProducer("JetFlavourClustering",
    jets                     = cms.InputTag("ca15PFJets"),
    groomedJets              = cms.InputTag("caHEPTopTagJets"),
    subjets                  = cms.InputTag("caHEPTopTagJets", "caTopSubJets"),
    bHadrons                 = cms.InputTag("selectedHadronsAndPartons","bHadrons"),
    cHadrons                 = cms.InputTag("selectedHadronsAndPartons","cHadrons"),
    partons                  = cms.InputTag("selectedHadronsAndPartons","algorithmicPartons"),
    leptons                  = cms.InputTag("selectedHadronsAndPartons","leptons"),
    jetAlgorithm             = cms.string("CambridgeAachen"),
    rParam                   = cms.double(1.5),
    ghostRescaling           = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(False)
)

process.printEventCA15PFJets = cms.EDAnalyzer("printJetFlavourInfo",
    jetFlavourInfos    = cms.InputTag("jetFlavourInfosCA15PFJets"),
    subjetFlavourInfos = cms.InputTag("jetFlavourInfosCA15PFJets","SubJets"),
    groomedJets        = cms.InputTag("caHEPTopTagJets"),
)
#-------------------------------------------------------------------------

process.p = cms.Path(
    process.printList
    *process.selectedHadronsAndPartons
    *process.jetFlavourInfosAK5PFJets*process.printEventAK5PFJets
    *(process.ak8PFJets+process.ak8PFJetsPruned)*process.jetFlavourInfosAK8PFJets*process.printEventAK8PFJets
    *(process.ca15PFJets+process.caHEPTopTagJets)*process.jetFlavourInfosCA15PFJets*process.printEventCA15PFJets
)

process.MessageLogger.destinations = cms.untracked.vstring('cout','cerr')
#process.MessageLogger.cout = cms.PSet(
#    threshold = cms.untracked.string('ERROR')
#)

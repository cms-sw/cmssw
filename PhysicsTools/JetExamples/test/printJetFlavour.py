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
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_5_3_11_patch3/RelValTTbar/GEN-SIM-RECO/PU_START53_LV2_09Jul2013-v1/00000/A26AB2F9-41E9-E211-806F-002590593920.root')
)

process.printTree = cms.EDAnalyzer("ParticleListDrawer",
    src = cms.InputTag("genParticles"),
    maxEventsToPrint  = cms.untracked.int32(1)
)

process.myPartons = cms.EDProducer("PartonSelector",
    src = cms.InputTag("genParticles"),
    withLeptons = cms.bool(False)
)

process.flavourByRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("ak5PFJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

process.flavourByVal = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("flavourByRef"),
    physicsDefinition = cms.bool(False)
)

process.printEvent = cms.EDAnalyzer("printJetFlavour",
    srcSelectedPartons = cms.InputTag("myPartons"),
    srcByReference = cms.InputTag("flavourByRef"),
    srcByValue = cms.InputTag("flavourByVal")
)

process.p = cms.Path(process.printTree*process.myPartons*process.flavourByRef*process.flavourByVal*process.printEvent)

process.MessageLogger.destinations = cms.untracked.vstring('cout','cerr')
#process.MessageLogger.cout = cms.PSet(
#    threshold = cms.untracked.string('ERROR')
#)



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

process.p = cms.Path(process.printList*process.myPartons*process.flavourByRef*process.flavourByVal*process.printEvent)

#process.MessageLogger.cout = dict(enable = True, threshold = 'ERROR')



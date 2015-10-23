import FWCore.ParameterSet.Config as cms

patJetPartonsLegacy = cms.EDProducer("PartonSelector",
    withLeptons = cms.bool(False),
    src = cms.InputTag("genParticles")
)

patJetPartonAssociationLegacy = cms.EDProducer("JetPartonMatcher",
    jets    = cms.InputTag("ak4PFJetsCHS"),
    partons = cms.InputTag("patJetPartonsLegacy"),
    coneSizeToAssociate = cms.double(0.3),
)

patJetFlavourAssociationLegacy = cms.EDProducer("JetFlavourIdentifier",
    srcByReference    = cms.InputTag("patJetPartonAssociationLegacy"),
    physicsDefinition = cms.bool(False)
)

patJetPartons = cms.EDProducer('HadronAndPartonSelector',
    src = cms.InputTag("generatorSmeared"),
    particles = cms.InputTag("genParticles"),
    partonMode = cms.string("Auto")
)

patJetFlavourAssociation = cms.EDProducer("JetFlavourClustering",
    jets = cms.InputTag("ak4PFJetsCHS"),
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    partons = cms.InputTag("patJetPartons","algorithmicPartons"),
    leptons = cms.InputTag("patJetPartons","leptons"),
    jetAlgorithm = cms.string("AntiKt"),
    rParam = cms.double(0.4),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(False)
)

# default PAT sequence for jet flavour identification
patJetFlavourIdLegacy = cms.Sequence(patJetPartonsLegacy * patJetPartonAssociationLegacy * patJetFlavourAssociationLegacy)
patJetFlavourId = cms.Sequence(patJetPartons * patJetFlavourAssociation)

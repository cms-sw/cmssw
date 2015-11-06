import FWCore.ParameterSet.Config as cms

patJetPartonsLegacy = cms.EDProducer("PartonSelector",
    withLeptons = cms.bool(False),
    src = cms.InputTag("genParticles")
)

patJetPartonAssociationLegacy = cms.EDProducer("JetPartonMatcher",
    jets    = cms.InputTag("ak5CaloJets"),
    partons = cms.InputTag("patJetPartonsLegacy"),
    coneSizeToAssociate = cms.double(0.3),
)

patJetFlavourAssociationLegacy = cms.EDProducer("JetFlavourIdentifier",
    srcByReference    = cms.InputTag("patJetPartonAssociationLegacy"),
    physicsDefinition = cms.bool(False)
)

patJetPartons = cms.EDProducer('HadronAndPartonSelector',
    src = cms.InputTag("generator"),
    particles = cms.InputTag("genParticles"),
    partonMode = cms.string("Auto")
)

patJetFlavourAssociation = cms.EDProducer("JetFlavourClustering",
    jets = cms.InputTag("ak5PFJets"),
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    partons = cms.InputTag("patJetPartons","partons"),
    jetAlgorithm = cms.string("AntiKt"),
    rParam = cms.double(0.5),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(True)
)

# default PAT sequence for jet flavour identification
patJetFlavourIdLegacy = cms.Sequence(patJetPartonsLegacy * patJetPartonAssociationLegacy * patJetFlavourAssociationLegacy)
patJetFlavourId = cms.Sequence(patJetPartons
                               #* patJetFlavourAssociation  ## new jet flavour disabled by default in 5_3_X
                               )

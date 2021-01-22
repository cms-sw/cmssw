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
    src = cms.InputTag("generator"),
    particles = cms.InputTag("genParticles"),
    partonMode = cms.string("Auto"),
    fullChainPhysPartons = cms.bool(True)
)

patJetFlavourAssociation = cms.EDProducer("JetFlavourClustering",
    jets = cms.InputTag("ak4PFJetsCHS"),
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    partons = cms.InputTag("patJetPartons","physicsPartons"),
    leptons = cms.InputTag("patJetPartons","leptons"),
    jetAlgorithm = cms.string("AntiKt"),
    rParam = cms.double(0.4),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(False)
)

# default PAT sequence for jet flavour identification
patJetFlavourIdLegacyTask = cms.Task(patJetPartonsLegacy, patJetPartonAssociationLegacy, patJetFlavourAssociationLegacy)
patJetFlavourIdLegacy = cms.Sequence(patJetFlavourIdLegacyTask)

patJetFlavourIdTask = cms.Task(patJetPartons, patJetFlavourAssociation)
patJetFlavourId = cms.Sequence(patJetFlavourIdTask)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(patJetPartons, particles = "hiSignalGenParticles")
pp_on_AA.toModify(patJetFlavourAssociation,
                  jets = 'akCs4PFJets',
                  rParam = 0.4,
)
pp_on_AA.toModify(patJetPartonAssociationLegacy,
                  jets = "akCs4PFJets",
                  partons = "allPartons",
)

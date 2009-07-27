import FWCore.ParameterSet.Config as cms

allLayer1Jets = cms.EDProducer("PATJetProducer",
    # input
    jetSource = cms.InputTag("iterativeCone5CaloJets"),
                               
    # add user data
    userData = cms.PSet(
      # add custom classes here
      userClasses = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add doubles here
      userFloats = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add ints here
      userInts = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add candidate ptrs here
      userCands = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add "inline" functions here
      userFunctions = cms.vstring(),
      userFunctionLabels = cms.vstring()
    ),
    
    # embedding of AOD items
    embedCaloTowers = cms.bool(False), ## switch on/off embedding of supercluster (externally stored in AOD)

    # jet energy corrections
    addJetCorrFactors    = cms.bool(True),
    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("jetCorrFactors") ),

    # btag information
    addBTagInfo          = cms.bool(True),   ## master switch
    addDiscriminators    = cms.bool(True),   ## addition btag discriminators
    discriminatorSources = cms.VInputTag(
        cms.InputTag("combinedSecondaryVertexBJetTags"),
        cms.InputTag("combinedSecondaryVertexMVABJetTags"),
        cms.InputTag("jetBProbabilityBJetTags"),
        cms.InputTag("jetProbabilityBJetTags"),
        cms.InputTag("simpleSecondaryVertexBJetTags"),
        cms.InputTag("softElectronByPtBJetTags"),                
        cms.InputTag("softElectronByIP3dBJetTags"),
        cms.InputTag("softMuonBJetTags"),
        cms.InputTag("softMuonByPtBJetTags"),                
        cms.InputTag("softMuonByIP3dBJetTags"),
        cms.InputTag("trackCountingHighEffBJetTags"),
        cms.InputTag("trackCountingHighPurBJetTags"),
    ),
    # clone tag infos ATTENTION: these take lots of space!
    # usually the discriminators from the default algos
    # are sufficient
    addTagInfos     = cms.bool(True),
    tagInfoSources  = cms.VInputTag(
        cms.InputTag("secondaryVertexTagInfos"),
        cms.InputTag("softElectronTagInfos"), 
        cms.InputTag("softMuonTagInfos"),
        cms.InputTag("impactParameterTagInfos"),
    ),

    # track association
    addAssociatedTracks    = cms.bool(True),
    trackAssociationSource = cms.InputTag("ic5JetTracksAssociatorAtVertex"),

    # jet charge
    addJetCharge    = cms.bool(True),
    jetChargeSource = cms.InputTag("patJetCharge"),

    # add jet ID
    addJetID = cms.bool(True),
    jetID = cms.PSet(
        hbheRecHitsColl = cms.InputTag("hbhereco"),
        hoRecHitsColl   = cms.InputTag("horeco"),
        hfRecHitsColl   = cms.InputTag("hfreco"),
        ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
        eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
    ),

    # mc matching
    addGenPartonMatch   = cms.bool(True),                 ## switch on/off matching to quarks from hard scatterin
    embedGenPartonMatch = cms.bool(False),                ## switch on/off embedding of the GenParticle parton for this jet
    genPartonMatch      = cms.InputTag("jetPartonMatch"), ## particles source to be used for the matching
    addGenJetMatch      = cms.bool(True),                 ## switch on/off matching to GenJet's
    genJetMatch         = cms.InputTag("jetGenJetMatch"), ## GenJet source to be used for the matching
    addPartonJetMatch   = cms.bool(False),                ## switch on/off matching to PartonJet's (not implemented yet)
    partonJetSource     = cms.InputTag("NOT_IMPLEMENTED"),## ParticleJet source to be used for the matching

    # jet flavour idetification configurables
    getJetMCFlavour    = cms.bool(True),
    JetPartonMapSource = cms.InputTag("jetFlavourAssociation"),

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution
    addResolutions = cms.bool(False),
    resolutions     = cms.PSet()
)



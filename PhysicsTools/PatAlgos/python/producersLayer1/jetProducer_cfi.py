import FWCore.ParameterSet.Config as cms

_patJets = cms.EDProducer("PATJetProducer",
    # input
    jetSource = cms.InputTag("ak4PFJetsCHS"),
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
    # embedding of RECO items (do not use on AOD input!)
    #embedCaloTowers = cms.bool(False), # optional
    # embedding of AOD items
    embedPFCandidates = cms.bool(False),
    # jet energy corrections
    addJetCorrFactors    = cms.bool(True),
    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactors") ),
    # btag information
    addBTagInfo          = cms.bool(True),   ## master switch
    addDiscriminators    = cms.bool(True),   ## addition btag discriminators
    discriminatorSources = cms.VInputTag(
        cms.InputTag("pfJetBProbabilityBJetTags"),
        cms.InputTag("pfJetProbabilityBJetTags"),
        cms.InputTag("pfTrackCountingHighEffBJetTags"),
        cms.InputTag("pfSimpleSecondaryVertexHighEffBJetTags"),
        cms.InputTag("pfSimpleInclusiveSecondaryVertexHighEffBJetTags"),
        cms.InputTag("pfCombinedSecondaryVertexV2BJetTags"),
        cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
        cms.InputTag("softPFMuonBJetTags"),
        cms.InputTag("softPFElectronBJetTags"),
        cms.InputTag("pfCombinedMVAV2BJetTags"),
        # CTagging
        cms.InputTag('pfCombinedCvsLJetTags'),
        cms.InputTag('pfCombinedCvsBJetTags'),
        # DeepFlavour
        cms.InputTag('pfDeepCSVJetTags:probb'),
        cms.InputTag('pfDeepCSVJetTags:probc'),
        cms.InputTag('pfDeepCSVJetTags:probudsg'),
        cms.InputTag('pfDeepCSVJetTags:probbb'),
        # New DeepFlavour (commented until available in RelVals)
        #cms.InputTag('pfDeepFlavourJetTags:probb'),
        #cms.InputTag('pfDeepFlavourJetTags:probbb'),
        #cms.InputTag('pfDeepFlavourJetTags:problepb'),
        #cms.InputTag('pfDeepFlavourJetTags:probc'),
        #cms.InputTag('pfDeepFlavourJetTags:probuds'),
        #cms.InputTag('pfDeepFlavourJetTags:probg')
    ),
    # clone tag infos ATTENTION: these take lots of space!
    # usually the discriminators from the default algos
    # are sufficient
    addTagInfos     = cms.bool(False),
    tagInfoSources  = cms.VInputTag(),
    # track association
    addAssociatedTracks    = cms.bool(True),
    trackAssociationSource = cms.InputTag("ak4JetTracksAssociatorAtVertexPF"),
    # jet charge
    addJetCharge    = cms.bool(True),
    jetChargeSource = cms.InputTag("patJetCharge"),
    # add jet ID for calo jets. This should be of type ak4JetID, ak7JetID, ...
    addJetID = cms.bool(False),
    jetIDMap = cms.InputTag("ak4JetID"),
    # mc matching
    addGenPartonMatch   = cms.bool(True),                           ## switch on/off matching to quarks from hard scatterin
    embedGenPartonMatch = cms.bool(True),                           ## switch on/off embedding of the GenParticle parton for this jet
    genPartonMatch      = cms.InputTag("patJetPartonMatch"),        ## particles source to be used for the matching
    addGenJetMatch      = cms.bool(True),                           ## switch on/off matching to GenJet's
    embedGenJetMatch    = cms.bool(True),                           ## switch on/off embedding of matched genJet's
    genJetMatch         = cms.InputTag("patJetGenJetMatch"),        ## GenJet source to be used for the matching
    addPartonJetMatch   = cms.bool(False),                          ## switch on/off matching to PartonJet's (not implemented yet)
    partonJetSource     = cms.InputTag("NOT_IMPLEMENTED"),          ## ParticleJet source to be used for the matching
    # jet flavour idetification configurables
    getJetMCFlavour    = cms.bool(True),
    useLegacyJetMCFlavour = cms.bool(False),
    addJetFlavourInfo  = cms.bool(True),
    JetPartonMapSource = cms.InputTag("patJetFlavourAssociationLegacy"),
    JetFlavourInfoSource = cms.InputTag("patJetFlavourAssociation"),
    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),
    # resolution
    addResolutions = cms.bool(False),
    resolutions     = cms.PSet()
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(_patJets, 
                                           jetSource = "akCs4PFJets",
                                           genJetMatch = "patJetGenJetMatch",
                                           genPartonMatch = "patJetPartonMatch",
                                           JetFlavourInfoSource = "patJetFlavourAssociation",
                                           JetPartonMapSource = "patJetFlavourAssociationLegacy",
                                           jetCorrFactorsSource = ["patJetCorrFactors"],
                                           trackAssociationSource = "ak5JetTracksAssociatorAtVertex",
                                           useLegacyJetMCFlavour = True,
                                           discriminatorSources = [
                                               "simpleSecondaryVertexHighEffBJetTags",
                                               "simpleSecondaryVertexHighPurBJetTags",
                                               "combinedSecondaryVertexV2BJetTags",
                                               "jetBProbabilityBJetTags",
                                               "jetProbabilityBJetTags",
                                               "trackCountingHighEffBJetTags",
                                               "trackCountingHighPurBJetTags",
                                           ],
                                           addJetCharge = False,
)

patJets = _patJets.clone()

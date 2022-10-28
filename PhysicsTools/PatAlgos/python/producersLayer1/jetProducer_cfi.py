import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.PATJetProducer_cfi as _mod

_patJets = _mod.PATJetProducer.clone(
    # input
    jetSource = "ak4PFJetsCHS",
    # add user data
    userData = dict(
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
      userFunctions = [],
      userFunctionLabels = []
    ),
    # embedding of RECO items (do not use on AOD input!)
    #embedCaloTowers = cms.bool(False), # optional
    # embedding of AOD items
    embedPFCandidates = False,
    # jet energy corrections
    addJetCorrFactors    = True,
    jetCorrFactorsSource = ["patJetCorrFactors" ],
    # btag information
    addBTagInfo          = True,   ## master switch
    addDiscriminators    = True,   ## addition btag discriminators
    discriminatorSources = ["pfJetBProbabilityBJetTags",
                            "pfJetProbabilityBJetTags",
                            "pfTrackCountingHighEffBJetTags",
                            "pfSimpleSecondaryVertexHighEffBJetTags",
                            "pfSimpleInclusiveSecondaryVertexHighEffBJetTags",
                            "pfCombinedSecondaryVertexV2BJetTags",
                            "pfCombinedInclusiveSecondaryVertexV2BJetTags",
                            "softPFMuonBJetTags",
                            "softPFElectronBJetTags",
                            "pfCombinedMVAV2BJetTags",
                            # CTagging
                            'pfCombinedCvsLJetTags',
                            'pfCombinedCvsBJetTags',
                            # DeepFlavour
                            'pfDeepCSVJetTags:probb',
                            'pfDeepCSVJetTags:probc',
                            'pfDeepCSVJetTags:probudsg',
                            'pfDeepCSVJetTags:probbb',
                            # New DeepFlavour (commented until available in RelVals)
                            #'pfDeepFlavourJetTags:probb',
                            #'pfDeepFlavourJetTags:probbb',
                            #'pfDeepFlavourJetTags:problepb',
                            #'pfDeepFlavourJetTags:probc',
                            #'pfDeepFlavourJetTags:probuds',
                            #'pfDeepFlavourJetTags:probg'
                           ],
    # clone tag infos ATTENTION: these take lots of space!
    # usually the discriminators from the default algos
    # are sufficient
    addTagInfos     = False,
    tagInfoSources  = [],
    # track association
    addAssociatedTracks    = True,
    trackAssociationSource = "ak4JetTracksAssociatorAtVertexPF",
    # jet charge
    addJetCharge    = True,
    jetChargeSource = "patJetCharge",
    # add jet ID for calo jets. This should be of type ak4JetID, ak7JetID, ...
    addJetID = False,
    jetIDMap = "ak4JetID",
    # mc matching
    addGenPartonMatch   = True,                ## switch on/off matching to quarks from hard scatterin
    embedGenPartonMatch = True,                ## switch on/off embedding of the GenParticle parton for this jet
    genPartonMatch      = "patJetPartonMatch", ## particles source to be used for the matching
    addGenJetMatch      = True,                ## switch on/off matching to GenJet's
    embedGenJetMatch    = True,                ## switch on/off embedding of matched genJet's
    genJetMatch         = "patJetGenJetMatch", ## GenJet source to be used for the matching
    addPartonJetMatch   = False,               ## switch on/off matching to PartonJet's (not implemented yet)
    partonJetSource     = "NOT_IMPLEMENTED",   ## ParticleJet source to be used for the matching
    # jet flavour idetification configurables
    getJetMCFlavour       = True,
    useLegacyJetMCFlavour = False,
    addJetFlavourInfo     = True,
    JetPartonMapSource    = "patJetFlavourAssociationLegacy",
    JetFlavourInfoSource  = "patJetFlavourAssociation",
    # efficiencies
    addEfficiencies = False,
    efficiencies    = dict(),
    # resolution
    addResolutions  = False,
    resolutions     = dict()
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(_patJets, 
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

from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
pp_on_PbPb_run3.toModify(_patJets, 
                         jetSource = "akCs4PFJets",
                         genJetMatch = "patJetGenJetMatch",
                         genPartonMatch = "patJetPartonMatch",
                         JetFlavourInfoSource = "patJetFlavourAssociation",
                         JetPartonMapSource = "patJetFlavourAssociationLegacy",
                         jetCorrFactorsSource = ["patJetCorrFactors"],
                         trackAssociationSource = "",
                         useLegacyJetMCFlavour = True,
                         discriminatorSources = [],
                         tagInfoSources = [],
                         addJetCharge = False,
                         addTagInfos = False,
                         addDiscriminators = False,
                         addAssociatedTracks    = False,
)

patJets = _patJets.clone()

import FWCore.ParameterSet.Config as cms

allLayer1Jets = cms.EDProducer("PATJetProducer",
    # General configurables
    jetSource = cms.InputTag("allLayer0Jets"),

                               
    # user data to add
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
      # add "inline" functions here
      userFunctions = cms.vstring(""),
      userFunctionLabels = cms.vstring("")
    ),
    
    # Embedding of AOD items
    embedCaloTowers = cms.bool(False), ## switch on/off embedding of supercluster (externally stored in AOD)

    # Jet Energy Corrections to appy and store
    addJetCorrFactors    = cms.bool(True),
    jetCorrFactorsSource = cms.InputTag("layer0JetCorrFactors"), ## source of the valuemap containing the jet correction factors

    # resolution configurables
    addResolutions = cms.bool(False),

    # -- BTagging information ---
    addBTagInfo = cms.bool(True), # master switch
    # copy discriminators in the pat::Jet
    addDiscriminators   = cms.bool(True),   ## switch on/off the addition of the btag discriminators
    discriminatorModule = cms.InputTag("layer0BTags"), ## meta-module which provides the list of discriminators. DO NOT specify an instance label
    discriminatorNames  = cms.vstring('*'), ## name of the JetTags to keep ( '*' = all )
    # clone tag infos in the pat::Jet
    # watch out: these take lots of space!
    # usually the discriminators from the default algos suffice
    addTagInfoRefs = cms.bool(True),
    tagInfoModule  = cms.InputTag("layer0TagInfos"),
    tagInfoNames   = cms.vstring('secondaryVertexTagInfos','softElectronTagInfos','softMuonTagInfos','impactParameterTagInfos'),

    # track association configurables
    addAssociatedTracks    = cms.bool(True),
    trackAssociationSource = cms.InputTag("layer0JetTracksAssociator"), ## the track association parameter set

    # Jet charge configurables
    addJetCharge    = cms.bool(True),
    jetChargeSource = cms.InputTag("layer0JetCharge"), ## the jet charge values

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    # trigger primitive sources to be used for the matching
    trigPrimMatch = cms.VInputTag(cms.InputTag("jetTrigMatchHLT1ElectronRelaxed"), cms.InputTag("jetTrigMatchHLT2jet")),

    # MC matching configurables
    addGenPartonMatch = cms.bool(True),                 ## switch on/off matching to quarks from hard scatterin
    embedGenPartonMatch = cms.bool(False),              ## switch on/off embedding of the GenParticle parton for this jet
    genPartonMatch    = cms.InputTag("jetPartonMatch"), ## particles source to be used for the matching
    addGenJetMatch    = cms.bool(True),                 ## switch on/off matching to GenJet's
    genJetMatch       = cms.InputTag("jetGenJetMatch"), ## GenJet source to be used for the matching
    addPartonJetMatch = cms.bool(False),                ## switch on/off matching to PartonJet's (not implemented yet)
    partonJetSource   = cms.InputTag("NOT_IMPLEMENTED"),## ParticleJet source to be used for the matching

    # Jet flavour idetification configurables
    getJetMCFlavour    = cms.bool(True),
    JetPartonMapSource = cms.InputTag("jetFlavourAssociation"), ## the match-collection, produced by default from PATHighLevelReco.cff

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),
)



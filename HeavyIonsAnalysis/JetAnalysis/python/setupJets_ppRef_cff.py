import FWCore.ParameterSet.Config as cms

def candidateBtaggingMiniAOD(process, isMC = True, jetPtMin = 15, jetCorrLevels = ['L2Relative', 'L3Absolute'], doBtagging = False, labelR = "0"):
    # DeepNtuple settings
    jetR = 0.1*float(labelR)
    if labelR == "0": jetR = 0.4

    jetCorrectionsAK4 = ('AK4PFchs' if labelR == "0" else 'AK'+labelR+'PFchs', jetCorrLevels, 'None')


    if doBtagging:
        bTagInfos = [
            'pfDeepCSVTagInfos',
            'pfDeepFlavourTagInfos',
            'pfImpactParameterTagInfos',
            'pfInclusiveSecondaryVertexFinderTagInfos',
            'pfParticleTransformerAK4TagInfos',
            'pfUnifiedParticleTransformerAK4TagInfos'
        ]

        bTagDiscriminators = [
            'pfUnifiedParticleTransformerAK4JetTags:probb',
            'pfUnifiedParticleTransformerAK4JetTags:probbb',
            'pfUnifiedParticleTransformerAK4JetTags:probc',
            'pfUnifiedParticleTransformerAK4JetTags:probg',
            'pfUnifiedParticleTransformerAK4JetTags:problepb',
            'pfUnifiedParticleTransformerAK4JetTags:probu',
            'pfUnifiedParticleTransformerAK4JetTags:probd',
            'pfUnifiedParticleTransformerAK4JetTags:probs',
            'pfUnifiedParticleTransformerAK4JetTags:probtaup1h0p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaup1h1p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaup1h2p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaup3h0p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaup3h1p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaum1h0p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaum1h1p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaum1h2p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaum3h0p',
            'pfUnifiedParticleTransformerAK4JetTags:probtaum3h1p',
            'pfUnifiedParticleTransformerAK4JetTags:probele',
            'pfUnifiedParticleTransformerAK4JetTags:probmu',
            'pfUnifiedParticleTransformerAK4JetTags:ptcorr',
            'pfUnifiedParticleTransformerAK4JetTags:ptnu',
        ]
    else: 
        bTagInfos = ['None']
        bTagDiscriminators = ['None']

    # Create gen-level information
    if isMC:
        from RecoHI.HiJetAlgos.hiSignalParticleProducer_cfi import hiSignalParticleProducer as hiSignalGenParticles
        process.hiSignalGenParticles = hiSignalGenParticles.clone(
            src = "prunedGenParticles"
        )
        from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJets_cff import allPartons
        process.allPartons = allPartons.clone(
            src = 'hiSignalGenParticles'
        )

        # Define generator level jets without neutrinos
        from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets

        process.packedGenParticlesForJetsNoNu = cms.EDFilter("CandPtrSelector",
            src = cms.InputTag("packedGenParticles"),
            cut = cms.string("abs(pdgId) != 12 && abs(pdgId) != 14 && abs(pdgId) != 16")
        )

        setattr(process,"ak"+labelR+"GenJetsReclusterNoNu",
                ak4GenJets.clone(
                    src = 'packedGenParticlesForJetsNoNu',
                    rParam = jetR
                )
        )
         # We need to be careful not to override the previous genTask in case several different jet radii are defined in the forest configuration file
        if hasattr(process, "genTask"):
            process.genTask.add( getattr(process,"ak"+labelR+"GenJetsReclusterNoNu"))
        else:
            process.genTask = cms.Task(process.hiSignalGenParticles, process.allPartons, process.packedGenParticlesForJetsNoNu, getattr(process,"ak"+labelR+"GenJetsReclusterNoNu"))


    # Create unsubtracted reco jets

    from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import ak4PFJets
    setattr(process, "ak"+labelR+"PFUnsubJets", 
            ak4PFJets.clone(
                src = 'packedPFCandidates',
                jetPtMin = 5.,  # set lower than subtracted version
                rParam = jetR
            )
    )
    

    if isMC:
        from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
        from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
        process.selectedHadronsAndPartons = selectedHadronsAndPartons.clone(particles = "prunedGenParticles")
        setattr(process,"ak"+labelR+"PFFlavourInfos",
                ak4JetFlavourInfos.clone(
                    jets = "ak"+labelR+"PFJetsCHS",
                    partons = "selectedHadronsAndPartons:algorithmicPartons",
                    hadronFlavourHasPriority = True,
                    rParam = jetR
                )
        )
        process.genTask.add(process.selectedHadronsAndPartons)
        process.genTask.add(getattr(process,"ak"+labelR+"PFFlavourInfos"))

    matchedGenJets = ""
    if isMC:
        if labelR == "0": matchedGenJets = "slimmedGenJets"
        else: matchedGenJets = "ak"+labelR+"GenJetsReclusterNoNu"


    svSource = cms.InputTag("slimmedSecondaryVertices")

    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    addJetCollection(
        process,
        postfix            = "UnsubJets",
        labelName          = str("AK"+labelR+"PF"),
        jetSource          = cms.InputTag("ak"+labelR+"PFUnsubJets"),
        algo               = "ak", #name of algo must be in this format
        rParam             = jetR,
        pvSource           = cms.InputTag("offlineSlimmedPrimaryVertices"),
        pfCandidates       = cms.InputTag("packedPFCandidates"),
        svSource           = svSource,
        muSource           = cms.InputTag("slimmedMuons"),
        elSource           = cms.InputTag("slimmedElectrons"),
        getJetMCFlavour    = isMC,
        genJetCollection   = cms.InputTag(matchedGenJets),
        genParticles       = cms.InputTag("hiSignalGenParticles" if isMC else ""),
        jetCorrections     = ('AK4PFchs' if labelR=='0' else 'AK'+labelR+'PFchs',) + jetCorrectionsAK4[1:],
    )

    getattr(process,"patJetsAK"+labelR+"PFUnsubJets").useLegacyJetMCFlavour = False

    process.patAlgosToolsTask.add(getattr(process,"ak"+labelR+"PFUnsubJets"))

    # Create CHS subtracted reco jets
    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    addJetCollection(
        process,
        postfix            = "",
        labelName          = "AK"+labelR+"PFCHS",
        jetSource          = cms.InputTag("ak"+labelR+"PFJetsCHS"),
        algo               = "ak", #name of algo must be in this format
        rParam             = jetR,
        pvSource           = cms.InputTag("offlineSlimmedPrimaryVertices"),
        pfCandidates       = cms.InputTag("packedPFCandidates"),
        svSource           = svSource,
        muSource           = cms.InputTag("slimmedMuons"),
        elSource           = cms.InputTag("slimmedElectrons"),
        getJetMCFlavour    = isMC,
        genJetCollection   = cms.InputTag(matchedGenJets),
        genParticles       = cms.InputTag("hiSignalGenParticles" if isMC else ""),
        jetCorrections     = jetCorrectionsAK4,
    )
    if labelR == "0": getattr(process,"patJetsAK"+labelR+"PFCHS").embedPFCandidates = True # not working with CHS jet reclustering

    if not isMC:
        for label in ["patJetsAK"+labelR+"PFUnsubJets", "patJetsAK"+labelR+"PFCHS"]:
            getattr(process, label).addGenJetMatch = False
            getattr(process, label).addGenPartonMatch = False
            getattr(process, label).embedGenJetMatch = False
            getattr(process, label).embedGenPartonMatch = False
            getattr(process, label).genJetMatch = ""
            getattr(process, label).genPartonMatch = ""

    # left here for reference in case we want to move reclustering here
    from CommonTools.ParticleFlow.pfCHS_cff import pfCHS
    process.pfCHS = pfCHS.clone()
    from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJetsCHS
    setattr(process,"ak"+labelR+"PFJetsCHS",
            ak4PFJetsCHS.clone(
                src = "pfCHS",
                jetPtMin = jetPtMin,
                rParam = jetR
            )
    )
    for mod in ["pfCHS","ak"+labelR+"PFJetsCHS"]:
        process.patAlgosToolsTask.add(getattr(process, mod))

    # Create b-tagging sequence ----------------
    from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
    updateJetCollection(
        process,
        labelName = "AK"+labelR+"PFCHSBtag",
        jetSource = cms.InputTag("slimmedJets" if labelR == "0" else "patJetsAK"+labelR+"PFCHS"), 
        jetCorrections = jetCorrectionsAK4,
        pfCandidates = cms.InputTag('packedPFCandidates'),
        pvSource = cms.InputTag("offlineSlimmedPrimaryVertices"),
        svSource = svSource,
        muSource = cms.InputTag('slimmedMuons'),
        elSource = cms.InputTag('slimmedElectrons'),
        btagInfos = bTagInfos,
        btagDiscriminators = bTagDiscriminators,
        explicitJTA = False
    )

    setattr(process,"unsubUpdatedPatJetsAK"+labelR+"PFCHS",
            cms.EDProducer("JetMatcherDR",
                           source = cms.InputTag("updatedPatJets"+labelR+"PFCHSBtag"),
                           matched = cms.InputTag("patJetsAK"+labelR+"PFUnsubJets")
                       )
        )
    process.patAlgosToolsTask.add(getattr(process,"unsubUpdatedPatJetsAK"+labelR+"PFCHS"))

    if doBtagging:
        getattr(process,"pfUnifiedParticleTransformerAK4JetTagsAK"+labelR+"PFCHSBtag").model_path = 'RecoBTag/Combined/data/UParTAK4/HIN/V00/UParTAK4_PbPb_2023.onnx'
        getattr(process,"pfUnifiedParticleTransformerAK4TagInfosAK"+labelR+"PFCHSBtag").sort_cand_by_pt = True 
        getattr(process,"pfUnifiedParticleTransformerAK4TagInfosAK"+labelR+"PFCHSBtag").fix_lt_sorting = True

        if hasattr(process,'updatedPatJetsTransientCorrectedAK'+labelR+'PFCHSBtag'):
            getattr(process,'updatedPatJetsTransientCorrectedAK'+labelR+'PFCHSBtag').addTagInfos = True
            getattr(process,'updatedPatJetsTransientCorrectedAK'+labelR+'PFCHSBtag').addBTagInfo = True
        else:
            raise ValueError('I could not find updatedPatJetsTransientCorrected to embed the tagInfos, please check the cfg')

            # Remove PUPPI
        process.patAlgosToolsTask.remove(process.packedpuppi)
        process.patAlgosToolsTask.remove(process.packedpuppiNoLep)
        getattr(process,"pfInclusiveSecondaryVertexFinderTagInfosAK"+labelR+"PFCHSBtag").weights = ""
        for taginfo in [ "pfDeepFlavourTagInfosAK"+labelR+"PFCHSBtag", "pfParticleTransformerAK4TagInfosAK"+labelR+"PFCHSBtag", "pfUnifiedParticleTransformerAK4TagInfosAK"+labelR+"PFCHSBtag"]:
            getattr(process, taginfo).fallback_puppi_weight = True
            getattr(process, taginfo).fallback_vertex_association = True
            getattr(process, taginfo).puppi_value_map = ""

    # Match with unsubtracted jets
    setattr(process,"unsubAK"+labelR+"JetMap",
            getattr(process,"unsubUpdatedPatJetsAK"+labelR+"PFCHS").clone(
                source = "selectedUpdatedPatJetsAK"+labelR+"PFCHS"
            )
        )

    process.patAlgosToolsTask.add(getattr(process,"unsubAK"+labelR+"JetMap"))

    # Add extra b tagging algos
    from RecoBTag.ImpactParameter.pfJetProbabilityBJetTags_cfi import pfJetProbabilityBJetTags
    setattr(process,"pfJetProbabilityBJetTagsAK"+labelR+"PFCHSBtag",
            pfJetProbabilityBJetTags.clone(tagInfos = ["pfImpactParameterTagInfosAK"+labelR+"PFCHSBtag"])
        )
    if doBtagging:
        process.patAlgosToolsTask.add(getattr(process,"pfJetProbabilityBJetTagsAK"+labelR+"PFCHSBtag"))

    # Associate to forest sequence
    if isMC:
        process.forest.associate(process.genTask)
    process.forest.associate(process.patAlgosToolsTask)

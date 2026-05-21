import FWCore.ParameterSet.Config as cms

def candidateBtaggingMiniAOD(process, isMC = True, jetPtMin = 15, jetCorrLevels = ['L2Relative', 'L3Absolute'], doBtagging = False, labelR = "0", iterativeFlow = True):

    # First, check if the label contains the string "Flow" to mark the use of flow subtraction
    doFlow = "Flow" in labelR

    # Determine the jet radius from the label
    jetR = 0.1*float(labelR.replace("Flow",""))
    if labelR == "0": jetR = 0.4

    #jetCorrectionsAK4 = ('AK4PF' if labelR == "0" else 'AK'+labelR+'PF', jetCorrLevels, 'None')
    jetCorrectionsAK4 = ('AK4PF', jetCorrLevels, 'None')  # temporary while we wait for updated JECs


    # DeepNtuple settings
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

        # Reclusted generator level jets without neutrinos
        from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
        process.packedGenParticlesForJetsNoNu = cms.EDFilter("CandPtrSelector",
            src = cms.InputTag("packedGenParticlesSignal"),
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
            process.genTask.add(getattr(process,"ak"+labelR+"GenJetsReclusterNoNu")) 
        else:
            process.genTask = cms.Task(process.hiSignalGenParticles, process.allPartons, process.packedGenParticlesForJetsNoNu, getattr(process,"ak"+labelR+"GenJetsReclusterNoNu"))

    # Remake secondary vertices
    from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import inclusiveCandidateVertexFinder, candidateVertexMerger, candidateVertexArbitrator, inclusiveCandidateSecondaryVertices
    process.inclusiveCandidateVertexFinder = inclusiveCandidateVertexFinder.clone(
        tracks = "packedPFCandidates",
        primaryVertices = "offlineSlimmedPrimaryVertices",
        minHits = 0,
        minPt = 0.8
    )
    process.candidateVertexMerger = candidateVertexMerger.clone()
    process.candidateVertexArbitrator = candidateVertexArbitrator.clone(
        tracks = "packedPFCandidates",
        primaryVertices = "offlineSlimmedPrimaryVertices"
    )
    process.inclusiveCandidateSecondaryVertices = inclusiveCandidateSecondaryVertices.clone()
    process.svTask = cms.Task(process.inclusiveCandidateVertexFinder, process.candidateVertexMerger, process.candidateVertexArbitrator, process.inclusiveCandidateSecondaryVertices)
    svSource = cms.InputTag("inclusiveCandidateSecondaryVertices")

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
        setattr(process,"ak"+labelR+"PFUnsubJetFlavourInfos",
                ak4JetFlavourInfos.clone(
                    jets = "ak"+labelR+"PFUnsubJets",
                    partons = "selectedHadronsAndPartons:algorithmicPartons",
                    hadronFlavourHasPriority = True,
                    rParam = jetR
                )
        )
        process.genTask.add(process.selectedHadronsAndPartons)
        process.genTask.add(getattr(process,"ak"+labelR+"PFUnsubJetFlavourInfos"))

    matchedGenJets = ""
    if isMC:
        if labelR == "0": matchedGenJets = "slimmedGenJets"
        else: matchedGenJets  = "ak"+labelR+"GenJetsReclusterNoNu"


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
        #jetCorrections     = ('AK4PF' if labelR=='0' else 'AK'+labelR+'PF',) + jetCorrectionsAK4[1:],
        jetCorrections     = ('AK4PF',) + jetCorrectionsAK4[1:],  #tempoorary while we wait for updated JECs
    )

    getattr(process,"patJetsAK"+labelR+"PFUnsubJets").useLegacyJetMCFlavour = False

    process.patAlgosToolsTask.add(getattr(process,"ak"+labelR+"PFUnsubJets"))

    # Create HIN subtracted reco jets
    addJetCollection(
        process,
        postfix            = "",
        labelName          = "AKCs"+labelR+"PF",
        jetSource          = cms.InputTag("akCs"+labelR+"PFJets"),
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
    getattr(process,"patJetsAKCs"+labelR+"PF").embedPFCandidates = True

    if not isMC:
        for label in ["patJetsAK"+labelR+"PFUnsubJets", "patJetsAKCs"+labelR+"PF"]:
            getattr(process, label).addGenJetMatch = False
            getattr(process, label).addGenPartonMatch = False
            getattr(process, label).embedGenJetMatch = False
            getattr(process, label).embedGenPartonMatch = False
            getattr(process, label).genJetMatch = ""
            getattr(process, label).genPartonMatch = ""

    # left here for reference in case we want to move reclustering here
    from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJets_cff import PackedPFTowers, hiPuRho
    process.PackedPFTowers = PackedPFTowers.clone()
    process.hiPuRho = hiPuRho.clone(
        src = 'PackedPFTowers'
    )

    # Define a list of modules that will need to be added to the process to cluster the jets
    processAdditives = ["PackedPFTowers", "hiPuRho"]

    # If we do flow subtraction, we need to setup the producers for flow modulation
    from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import akCs4PFJets
    iterativeTag = ""
    if doFlow:
        from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import ak4PFJetsForFlow, hiFJRhoFlowModulation
        setattr(process, "ak4PFJetsFor"+labelR,
                ak4PFJetsForFlow.clone(
                    src = "PackedPFTowers"
                )
        )

        setattr(process, "rhoModulationAkCs"+labelR+"PFJets",
                hiFJRhoFlowModulation.clone(
                    jetTag = "ak4PFJetsFor"+labelR
                )
        )

        # Both of these processes are needed for flow subtracted jets
        processAdditives.append("ak4PFJetsFor"+labelR)
        processAdditives.append("rhoModulationAkCs"+labelR+"PFJets")

        # For iterative flow, we need to use the previously determined flow components to create a jet collection that will be used in jetty region subtraction
        if iterativeFlow:
            setattr(process, "akCs4PFJetsFor"+labelR,
                    akCs4PFJets.clone(
                        src = 'packedPFCandidates',
                        rParam = jetR,
                        jetPtMin = 40,
                        useModulatedRho = cms.bool(True),
                        rhoFlowFitParams = cms.InputTag("rhoModulationAkCs"+labelR+"PFJets", "rhoFlowFitParams")
                    )
            )

            setattr(process, "rhoModulationIterAkCs"+labelR+"PFJets",
                    hiFJRhoFlowModulation.clone(
                        jetTag = "akCs4PFJetsFor"+labelR,
                        doJettyExclusion = cms.bool(True)
                    )
            )
            
            # Both of these processes are needed for flow subtracted jets
            processAdditives.append("akCs4PFJetsFor"+labelR)
            processAdditives.append("rhoModulationIterAkCs"+labelR+"PFJets")

            # Add iterative tag to the jet collection
            iterativeTag = "Iter"


    setattr(process,"akCs"+labelR+"PFJets",
            akCs4PFJets.clone(
                src = 'packedPFCandidates',
                jetPtMin = jetPtMin,
                rParam = jetR,
                useModulatedRho = doFlow,
                rhoFlowFitParams = cms.InputTag("rhoModulation"+iterativeTag+"AkCs"+labelR+"PFJets","rhoFlowFitParams")
            )
    )

    # Also the jet collection needs to be added to the process
    processAdditives.append("akCs"+labelR+"PFJets")

    for mod in processAdditives:
        process.patAlgosToolsTask.add(getattr(process, mod))

    # Create b-tagging sequence ----------------
    from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
    updateJetCollection(
        process,
        labelName = "AK"+labelR+"PFBtag",
        jetSource = cms.InputTag("slimmedJets" if labelR == "0" else "patJetsAKCs"+labelR+"PF"), 
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

    setattr(process,"unsubUpdatedPatJetsAK"+labelR+"PF",
            cms.EDProducer("JetMatcherDR",
            source = cms.InputTag("updatedPatJetsAK"+labelR+"PFBtag"),                                                     
            matched = cms.InputTag("patJetsAK"+labelR+"PFUnsubJets")
                       )
        )
    process.patAlgosToolsTask.add(getattr(process,"unsubUpdatedPatJetsAK"+labelR+"PF"))

    if doBtagging:

        getattr(process,"pfUnifiedParticleTransformerAK4JetTagsAK"+labelR+"PFBtag").model_path = 'RecoBTag/Combined/data/UParTAK4/HIN/V00/UParTAK4_PbPb_2023.onnx'
        getattr(process,"pfUnifiedParticleTransformerAK4TagInfosAK"+labelR+"PFBtag").sort_cand_by_pt = True
        getattr(process,"pfUnifiedParticleTransformerAK4TagInfosAK"+labelR+"PFBtag").fix_lt_sorting = True

        if hasattr(process,'updatedPatJetsTransientCorrectedAK'+labelR+'PFBtag'):
            getattr(process,'updatedPatJetsTransientCorrectedAK'+labelR+'PFBtag').addTagInfos = True
            getattr(process,'updatedPatJetsTransientCorrectedAK'+labelR+'PFBtag').addBTagInfo = True
        else:
            raise ValueError('I could not find updatedPatJetsTransientCorrected to embed the tagInfos, please check the cfg')

            # Remove PUPPI
        process.patAlgosToolsTask.remove(process.packedpuppi)
        process.patAlgosToolsTask.remove(process.packedpuppiNoLep)
        getattr(process,"pfInclusiveSecondaryVertexFinderTagInfosAK"+labelR+"PFBtag").weights = ""
        for taginfo in [ "pfDeepFlavourTagInfosAK"+labelR+"PFBtag", "pfParticleTransformerAK4TagInfosAK"+labelR+"PFBtag", "pfUnifiedParticleTransformerAK4TagInfosAK"+labelR+"PFBtag"]:
            getattr(process, taginfo).fallback_puppi_weight = True
            getattr(process, taginfo).fallback_vertex_association = True
            getattr(process, taginfo).unsubjet_map = "unsubUpdatedPatJetsAK"+labelR+"PF"
            getattr(process, taginfo).puppi_value_map = ""
            
    # Match with unsubtracted jets                                                                                                                             
    setattr(process,"unsubAK"+labelR+"JetMap",
            getattr(process,"unsubUpdatedPatJetsAK"+labelR+"PF").clone(
                source = "selectedUpdatedPatJetsAK"+labelR+"PF"
            )
        )

    process.patAlgosToolsTask.add(getattr(process,"unsubAK"+labelR+"JetMap"))

    # Add extra b tagging algos
    from RecoBTag.ImpactParameter.pfJetProbabilityBJetTags_cfi import pfJetProbabilityBJetTags
    setattr(process,"pfJetProbabilityBJetTagsAK"+labelR+"PFBtag",
            pfJetProbabilityBJetTags.clone(tagInfos = ["pfImpactParameterTagInfosAK"+labelR+"PFBtag"])
        )

    if doBtagging:
        process.patAlgosToolsTask.add(getattr(process,"pfJetProbabilityBJetTagsAK"+labelR+"PFBtag"))

    # Associate to forest sequence
    if isMC:
        process.forest.associate(process.genTask)
    if doBtagging: 
        process.forest.associate(process.svTask)
    process.forest.associate(process.patAlgosToolsTask)

import FWCore.ParameterSet.Config as cms

def applySubstructure( process ) :

    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

    # add CMS top tagger
    from RecoJets.JetProducers.caTopTaggers_cff import caTopTagInfos
    process.caTopTagInfos = caTopTagInfos.clone()
    process.caTopTagInfosPAT = cms.EDProducer("RecoJetDeltaRTagInfoValueMapProducer",
                                    src = cms.InputTag("ak8PFJetsCHS"),
                                    matched = cms.InputTag("cmsTopTagPFJetsCHS"),
                                    matchedTagInfos = cms.InputTag("caTopTagInfos"),
                                    distMax = cms.double(0.8)
    )    
    
    #add AK8
    addJetCollection(process, labelName = 'AK8',
                     jetSource = cms.InputTag('ak8PFJetsCHS'),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     btagInfos = ['caTopTagInfosPAT'],
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    process.patJetsAK8.userData.userFloats.src = [] # start with empty list of user floats
    process.selectedPatJetsAK8.cut = cms.string("pt > 50")



    ## AK8 groomed masses
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSPruned, ak8PFJetsCHSSoftDrop, ak8PFJetsCHSFiltered, ak8PFJetsCHSTrimmed 
    process.ak8PFJetsCHSPruned   = ak8PFJetsCHSPruned.clone()
    process.ak8PFJetsCHSSoftDrop = ak8PFJetsCHSSoftDrop.clone()
    process.ak8PFJetsCHSTrimmed  = ak8PFJetsCHSTrimmed.clone()
    process.ak8PFJetsCHSFiltered = ak8PFJetsCHSFiltered.clone()
    process.load("RecoJets.JetProducers.ak8PFJetsCHS_groomingValueMaps_cfi")
    process.patJetsAK8.userData.userFloats.src += ['ak8PFJetsCHSPrunedMass','ak8PFJetsCHSSoftDropMass','ak8PFJetsCHSTrimmedMass','ak8PFJetsCHSFilteredMass']

    # Add AK8 top tagging variables
    process.patJetsAK8.tagInfoSources = cms.VInputTag(cms.InputTag("caTopTagInfosPAT"))
    process.patJetsAK8.addTagInfos = cms.bool(True)



    # add Njetiness
    process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
    process.NjettinessAK8 = process.Njettiness.clone()
    process.NjettinessAK8.src = cms.InputTag("ak8PFJetsCHS")
    process.NjettinessAK8.cone = cms.double(0.8)
    process.patJetsAK8.userData.userFloats.src += ['NjettinessAK8:tau1','NjettinessAK8:tau2','NjettinessAK8:tau3']


    ## PATify pruned fat jets
    addJetCollection(
        process,
        labelName = 'AK8PFCHSSoftDrop',
        jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop'),
        btagDiscriminators = ['None'],
        jetCorrections = ('AK8PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        getJetMCFlavour = False # jet flavor disabled
    )
    getattr(process,'patJetPartonMatchAK8PFCHSSoftDrop').matched = cms.InputTag('prunedGenParticles')
    ## PATify soft drop subjets
    addJetCollection(
        process,
        labelName = 'AK8PFCHSSoftDropSubjets',
        jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop','SubJets'),
        algo = 'ak',  # needed for subjet flavor clustering
        rParam = 0.8, # needed for subjet flavor clustering
        btagDiscriminators = ['pfCombinedSecondaryVertexBJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags'],
        jetCorrections = ('AK4PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        explicitJTA = True,  # needed for subjet b tagging
        svClustering = True, # needed for subjet b tagging
        genJetCollection = cms.InputTag('slimmedGenJets'), 
        fatJets=cms.InputTag('ak8PFJetsCHS'),             # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('ak8PFJetsCHSSoftDrop') # needed for subjet flavor clustering
    )
    process.selectedPatJetsAK8PFCHSSoftDrop.cut = cms.string("pt > 200")
    
    process.slimmedJetsAK8PFCHSSoftDropSubjets = cms.EDProducer("PATJetSlimmer",
        src = cms.InputTag("selectedPatJetsAK8PFCHSSoftDropSubjets"),
        packedPFCandidates = cms.InputTag("packedPFCandidates"),
        dropJetVars = cms.string("1"),
        dropDaughters = cms.string("0"),
        dropTrackRefs = cms.string("1"),
        dropSpecific = cms.string("1"),
        dropTagInfos = cms.string("1"),
    )

    
    ## Establish references between PATified fat jets and subjets using the BoostedJetMerger
    process.slimmedJetsAK8PFCHSSoftDropPacked = cms.EDProducer("BoostedJetMerger",
        jetSrc=cms.InputTag("selectedPatJetsAK8PFCHSSoftDrop"),
        subjetSrc=cms.InputTag("slimmedJetsAK8PFCHSSoftDropSubjets")
    )
    
    addJetCollection(
        process,
        labelName = 'CMSTopTagCHS',
        jetSource = cms.InputTag('cmsTopTagPFJetsCHS'),
        jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
        btagDiscriminators = ['pfCombinedSecondaryVertexBJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags'],
        genJetCollection = cms.InputTag('slimmedGenJetsAK8'), 
        getJetMCFlavour = False #
        )
    process.patJetsCMSTopTagCHS.addTagInfos = True
    process.patJetsCMSTopTagCHS.tagInfoSources = cms.VInputTag(
        cms.InputTag('caTopTagInfosPAT')
        )
    process.selectedPatJetsCMSTopTagCHS.cut = cms.string("pt > 200")

    addJetCollection(
        process,
        labelName = 'CMSTopTagCHSSubjets',
        jetSource = cms.InputTag('cmsTopTagPFJetsCHS','caTopSubJets'),
        algo = 'AK',  # needed for subjet flavor clustering
        rParam = 0.8, # needed for subjet flavor clustering        
        btagDiscriminators = ['pfCombinedSecondaryVertexBJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags'],
        jetCorrections = ('AK4PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        genJetCollection = cms.InputTag('slimmedGenJets'), # Using ak4GenJets for matching which is not entirely appropriate
        explicitJTA = True,  # needed for subjet b tagging
        svClustering = True, # needed for subjet b tagging
        fatJets=cms.InputTag('ak8PFJetsCHS'),             # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('cmsTopTagPFJetsCHS') # needed for subjet flavor clustering

        )

    process.slimmedJetsCMSTopTagCHSSubjets = cms.EDProducer("PATJetSlimmer",
        src = cms.InputTag("selectedPatJetsCMSTopTagCHSSubjets"),
        packedPFCandidates = cms.InputTag("packedPFCandidates"),
        dropJetVars = cms.string("1"),
        dropDaughters = cms.string("0"),
        dropTrackRefs = cms.string("1"),
        dropSpecific = cms.string("1"),
        dropTagInfos = cms.string("1"),
    )
    
    ## Establish references between PATified fat jets and subjets using the BoostedJetMerger
    process.slimmedJetsCMSTopTagCHSPacked = cms.EDProducer("BoostedJetMerger",
        jetSrc=cms.InputTag("selectedPatJetsCMSTopTagCHS"),
        subjetSrc=cms.InputTag("slimmedJetsCMSTopTagCHSSubjets")
    )


    process.packedPatJetsAK8 = cms.EDProducer("JetSubstructurePacker",
            jetSrc = cms.InputTag("selectedPatJetsAK8"),
            distMin = cms.double(0.8),
            algoTags = cms.VInputTag(
                cms.InputTag("slimmedJetsCMSTopTagCHSPacked"),
                cms.InputTag("slimmedJetsAK8PFCHSSoftDropPacked"),
            ),
            algoLabels = cms.vstring(
                'CMSTopTag',
                'SoftDrop',
                )
    )



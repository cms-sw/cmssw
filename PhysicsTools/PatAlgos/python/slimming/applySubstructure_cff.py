import FWCore.ParameterSet.Config as cms

def applySubstructure( process ) :

    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection


    from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets as patJetsDefault

    #add AK8
    addJetCollection(process, labelName = 'AK8',
                     jetSource = cms.InputTag('ak8PFJetsCHS'),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    process.patJetsAK8.userData.userFloats.src = [] # start with empty list of user floats
    process.selectedPatJetsAK8.cut = cms.string("pt > 170")


    ## AK8 groomed masses
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSPruned, ak8PFJetsCHSSoftDrop
    process.ak8PFJetsCHSPruned   = ak8PFJetsCHSPruned.clone()
    process.ak8PFJetsCHSSoftDrop = ak8PFJetsCHSSoftDrop.clone()
    process.load("RecoJets.JetProducers.ak8PFJetsCHS_groomingValueMaps_cfi")
    process.patJetsAK8.userData.userFloats.src += ['ak8PFJetsCHSPrunedMass','ak8PFJetsCHSSoftDropMass']  
    process.patJetsAK8.addTagInfos = cms.bool(False)



    # add Njetiness
    process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
    process.NjettinessAK8 = process.Njettiness.clone()
    process.NjettinessAK8.src = cms.InputTag("ak8PFJetsCHS")
    process.NjettinessAK8.cone = cms.double(0.8)
    process.patJetsAK8.userData.userFloats.src += ['NjettinessAK8:tau1','NjettinessAK8:tau2','NjettinessAK8:tau3']




    #add AK8 from PUPPI
    process.load('RecoJets.JetProducers.ak8PFJetsPuppi_cfi')
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsPuppiSoftDrop 
    process.ak8PFJetsPuppiSoftDrop = ak8PFJetsPuppiSoftDrop.clone()
    process.ak8PFJetsPuppi.doAreaFastjet = True # even for standard ak8PFJets this is overwritten in RecoJets/Configuration/python/RecoPFJets_cff

        
    addJetCollection(process, labelName = 'AK8Puppi',
                     jetSource = cms.InputTag('ak8PFJetsPuppi'),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFPuppi', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     btagDiscriminators = ([x.getModuleLabel() for x in patJetsDefault.discriminatorSources] + ['pfBoostedDoubleSecondaryVertexAK8BJetTags']),
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    process.patJetsAK8Puppi.userData.userFloats.src = [] # start with empty list of user floats
    process.selectedPatJetsAK8Puppi.cut = cms.string("pt > 170")


    from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import j2tParametersVX
    process.ak8PFJetsPuppiTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
        j2tParametersVX.clone( coneSize = cms.double(0.8) ),
        jets = cms.InputTag("ak8PFJetsPuppi")        
    )
    process.patJetAK8PuppiCharge = cms.EDProducer("JetChargeProducer",
        src = cms.InputTag("ak8PFJetsPuppiTracksAssociatorAtVertex"),
        var = cms.string('Pt'),
        exp = cms.double(1.0)
    )

    ## AK8 groomed masses
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsPuppiSoftDrop
    process.ak8PFJetsPuppiSoftDrop = ak8PFJetsPuppiSoftDrop.clone()
    process.load("RecoJets.JetProducers.ak8PFJetsPuppi_groomingValueMaps_cfi")
    process.patJetsAK8Puppi.userData.userFloats.src += ['ak8PFJetsPuppiSoftDropMass']
    process.patJetsAK8Puppi.addTagInfos = cms.bool(False)



    # add Njetiness
    process.NjettinessAK8Puppi = process.Njettiness.clone()
    process.NjettinessAK8Puppi.src = cms.InputTag("ak8PFJetsPuppi")
    process.NjettinessAK8Puppi.cone = cms.double(0.8)
    process.patJetsAK8Puppi.userData.userFloats.src += ['NjettinessAK8Puppi:tau1','NjettinessAK8Puppi:tau2','NjettinessAK8Puppi:tau3']




    process.ak8PFJetsCHSValueMap = cms.EDProducer("RecoJetToPatJetDeltaRValueMapProducer",
                                            src = cms.InputTag("ak8PFJetsPuppi"),
                                            matched = cms.InputTag("patJetsAK8"),
                                            distMax = cms.double(0.8),
                                            values = cms.vstring([
                                                'userFloat("ak8PFJetsCHSPrunedMass")',
                                                'userFloat("ak8PFJetsCHSSoftDropMass")',
                                                'userFloat("NjettinessAK8:tau1")',
                                                'userFloat("NjettinessAK8:tau2")',
                                                'userFloat("NjettinessAK8:tau3")',
                                                'pt','eta','phi','mass'
                                            ]),
                                            valueLabels = cms.vstring( [
                                                'ak8PFJetsCHSPrunedMass',
                                                'ak8PFJetsCHSSoftDropMass',
                                                'NjettinessAK8CHSTau1',
                                                'NjettinessAK8CHSTau2',
                                                'NjettinessAK8CHSTau3',
                                                'pt','eta','phi','mass'
                                            ])
                        )
    process.patJetsAK8Puppi.userData.userFloats.src += [
                                                   cms.InputTag('ak8PFJetsCHSValueMap','ak8PFJetsCHSPrunedMass'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap','ak8PFJetsCHSSoftDropMass'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap','NjettinessAK8CHSTau1'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap','NjettinessAK8CHSTau2'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap','NjettinessAK8CHSTau3'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap','pt'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap','eta'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap','phi'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap','mass'),
                                                   ]

    # add Njetiness
    process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
    process.NjettinessAK8Subjets = process.Njettiness.clone()
    process.NjettinessAK8Subjets.src = cms.InputTag("ak8PFJetsPuppiSoftDrop", "SubJets")
    process.NjettinessAK8Subjets.cone = cms.double(0.8)
    

    
    ## PATify CHS soft drop fat jets
    addJetCollection(
        process,
        labelName = 'AK8PFCHSSoftDrop',
        jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop'),
        btagDiscriminators = ['None'],
        jetCorrections = ('AK8PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        getJetMCFlavour = False # jet flavor disabled
    )


    ## PATify puppi soft drop fat jets
    addJetCollection(
        process,
        labelName = 'AK8PFPuppiSoftDrop',
        jetSource = cms.InputTag('ak8PFJetsPuppiSoftDrop'),
        btagDiscriminators = ['None'],
        jetCorrections = ('AK8PFPuppi', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        getJetMCFlavour = False # jet flavor disabled
    )
    
    ## PATify soft drop subjets
    addJetCollection(
        process,
        labelName = 'AK8PFPuppiSoftDropSubjets',
        jetSource = cms.InputTag('ak8PFJetsPuppiSoftDrop','SubJets'),
        algo = 'ak',  # needed for subjet flavor clustering
        rParam = 0.8, # needed for subjet flavor clustering
        btagDiscriminators = ['pfCombinedSecondaryVertexV2BJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags','pfCombinedMVAV2BJetTags'],
        jetCorrections = ('AK4PFPuppi', ['L2Relative', 'L3Absolute'], 'None'),
        explicitJTA = True,  # needed for subjet b tagging
        svClustering = True, # needed for subjet b tagging
        genJetCollection = cms.InputTag('slimmedGenJets'), 
        fatJets=cms.InputTag('ak8PFJetsPuppi'),             # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('ak8PFJetsPuppiSoftDrop') # needed for subjet flavor clustering
    )
    process.selectedPatJetsAK8PFPuppiSoftDrop.cut = cms.string("pt > 170")
    process.patJetsAK8PFPuppiSoftDropSubjets.userData.userFloats.src += ['NjettinessAK8Subjets:tau1','NjettinessAK8Subjets:tau2','NjettinessAK8Subjets:tau3']
    
    process.slimmedJetsAK8PFPuppiSoftDropSubjets = cms.EDProducer("PATJetSlimmer",
        src = cms.InputTag("selectedPatJetsAK8PFPuppiSoftDropSubjets"),
        packedPFCandidates = cms.InputTag("packedPFCandidates"),
        dropJetVars = cms.string("1"),
        dropDaughters = cms.string("0"),
        rekeyDaughters = cms.string("1"),
        dropTrackRefs = cms.string("1"),
        dropSpecific = cms.string("1"),
        dropTagInfos = cms.string("1"),
        modifyJets = cms.bool(True),
        modifierConfig = cms.PSet( modifications = cms.VPSet() )
    )

    
    ## Establish references between PATified fat jets and subjets using the BoostedJetMerger
    process.slimmedJetsAK8PFPuppiSoftDropPacked = cms.EDProducer("BoostedJetMerger",
        jetSrc=cms.InputTag("selectedPatJetsAK8PFPuppiSoftDrop"),
        subjetSrc=cms.InputTag("slimmedJetsAK8PFPuppiSoftDropSubjets")
    )

    
    process.packedPatJetsAK8 = cms.EDProducer("JetSubstructurePacker",
            jetSrc = cms.InputTag("selectedPatJetsAK8Puppi"),
            distMax = cms.double(0.8),
            algoTags = cms.VInputTag(
                cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked")
            ),
            algoLabels = cms.vstring(
                'SoftDropPuppi'
                ),
            fixDaughters = cms.bool(True),
            packedPFCandidates = cms.InputTag("packedPFCandidates"),
    )

    # switch off daughter re-keying since it's done in the JetSubstructurePacker (and can't be done afterwards)
    process.slimmedJetsAK8.rekeyDaughters = "0"


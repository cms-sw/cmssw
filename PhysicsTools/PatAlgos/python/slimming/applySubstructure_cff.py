import FWCore.ParameterSet.Config as cms

def applySubstructure( process, postfix="" ) :

    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection


    from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets as patJetsDefault

    #add AK8
    addJetCollection(process, postfix=postfix, labelName = 'AK8',
                     jetSource = cms.InputTag('ak8PFJetsCHS'+postfix),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     btagDiscriminators = ([x.getModuleLabel() for x in patJetsDefault.discriminatorSources] + ['pfBoostedDoubleSecondaryVertexAK8BJetTags']),
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    getattr(process, "patJetsAK8"+postfix).userData.userFloats.src = [] # start with empty list of user floats
    getattr(process,"selectedPatJetsAK8"+postfix).cut = cms.string("pt > 170")

    from RecoJets.JetProducers.ak8PFJetsPuppi_cfi import ak8PFJetsPuppi
    #process.load('RecoJets.JetProducers.ak8PFJetsPuppi_cfi')
    setattr(process, "ak8PFJetsPuppi"+postfix, ak8PFJetsPuppi.clone(
            doAreaFastjet=True)  # even for standard ak8PFJets this is overwritten in RecoJets/Configuration/python/RecoPFJets_cff
            )

    from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import j2tParametersVX
    setattr(process,"ak8PFJetsPuppiTracksAssociatorAtVertex"+postfix, 
            cms.EDProducer("JetTracksAssociatorAtVertex",
                           j2tParametersVX,
                           jets = cms.InputTag("ak8PFJetsPuppi"+postfix)
                           ) 
            )
    setattr(process, "patJetPuppiCharge"+postfix, 
            cms.EDProducer("JetChargeProducer",
                           src = cms.InputTag("ak8PFJetsPuppiTracksAssociatorAtVertex"+postfix),
                           var = cms.string('Pt'),
                           exp = cms.double(1.0)
                           ) 
            )

    ## AK8 groomed masses
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSPruned, ak8PFJetsCHSSoftDrop, ak8PFJetsPuppiSoftDrop 
    setattr(process, "ak8PFJetsCHSPruned"+postfix, ak8PFJetsCHSPruned.clone(
            src = cms.InputTag("pfNoPileUpJME"+postfix),
            ) )
    setattr(process, "ak8PFJetsCHSSoftDrop"+postfix, ak8PFJetsCHSSoftDrop.clone(
            src = cms.InputTag("pfNoPileUpJME"+postfix),
            ) )
    from RecoJets.JetProducers.ak8PFJetsCHS_groomingValueMaps_cfi import ak8PFJetsCHSPrunedMass, ak8PFJetsCHSSoftDropMass
    setattr(process, "ak8PFJetsCHSPrunedMass"+postfix, ak8PFJetsCHSPrunedMass.clone(
            src = cms.InputTag("ak8PFJetsCHS"+postfix),
            matched = cms.InputTag("ak8PFJetsCHSPruned"+postfix),
            ) )
    setattr(process, "ak8PFJetsCHSSoftDropMass"+postfix, ak8PFJetsCHSSoftDropMass.clone(
            src = cms.InputTag("ak8PFJetsCHS"+postfix),
            matched = cms.InputTag("ak8PFJetsCHSSoftDrop"+postfix), 
            ) )

    getattr(process, "patJetsAK8"+postfix).userData.userFloats.src += ['ak8PFJetsCHSPrunedMass'+postfix,'ak8PFJetsCHSSoftDropMass'+postfix] 
    getattr(process, "patJetsAK8"+postfix).addTagInfos = cms.bool(False)


    # add Njetiness
    from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness
    #process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
    setattr(process,"NjettinessAK8"+postfix, Njettiness.clone(
            src = cms.InputTag("ak8PFJetsCHS"+postfix),
            cone = cms.double(0.8) )
            )
    getattr(process, "patJetsAK8"+postfix).userData.userFloats.src += ['NjettinessAK8'+postfix+':tau1','NjettinessAK8'+postfix+':tau2','NjettinessAK8'+postfix+':tau3']




    #add AK8 from PUPPI
    #MM no need for that, already done on L23
    #process.load('RecoJets.JetProducers.ak8PFJetsPuppi_cfi')
    #from RecoJets.JetProducers.ak8PFJetsPuppi_cfi import ak8PFJetsPuppi
    #getattr(process, "ak8PFJetsPuppi".doAreaFastjet = True # even for standard ak8PFJets this is overwritten in RecoJets/Configuration/python/RecoPFJets_cff

        
    addJetCollection(process, postfix=postfix,labelName = 'AK8Puppi',
                     jetSource = cms.InputTag('ak8PFJetsPuppi'+postfix),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFPuppi', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src = [] # start with empty list of user floats
    getattr(process,"selectedPatJetsAK8Puppi"+postfix).cut = cms.string("pt > 170")


    from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import j2tParametersVX
    setattr(process,"ak8PFJetsPuppiTracksAssociatorAtVertex"+postfix,
            cms.EDProducer("JetTracksAssociatorAtVertex",
                           j2tParametersVX,
                           jets = cms.InputTag("ak8PFJetsPuppi"+postfix) )
            )
    setattr(process,"patJetAK8PuppiCharge"+postfix,
            cms.EDProducer("JetChargeProducer",
                           src = cms.InputTag("ak8PFJetsPuppiTracksAssociatorAtVertex"+postfix),
                           var = cms.string('Pt'),
                           exp = cms.double(1.0) )
            )

    ## AK8 groomed masses
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsPuppiSoftDrop
    setattr(process,"ak8PFJetsPuppiSoftDrop"+postfix, ak8PFJetsPuppiSoftDrop.clone(
            src = cms.InputTag("puppi"+postfix),
            ) )
    from RecoJets.JetProducers.ak8PFJetsPuppi_groomingValueMaps_cfi import ak8PFJetsPuppiSoftDropMass
    setattr(process, "ak8PFJetsPuppiSoftDropMass"+postfix, ak8PFJetsPuppiSoftDropMass.clone(
            src = cms.InputTag("ak8PFJetsPuppi"+postfix),
            matched = cms.InputTag("ak8PFJetsPuppiSoftDrop"+postfix), 
            ) )
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src += ['ak8PFJetsPuppiSoftDropMass'+postfix]
    getattr(process,"patJetsAK8Puppi"+postfix).addTagInfos = cms.bool(False)



    # add Njetiness
    setattr(process,"NjettinessAK8Puppi"+postfix, Njettiness.clone(
            src = cms.InputTag("ak8PFJetsPuppi"+postfix),
            cone = cms.double(0.8) )
            )
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src += ['NjettinessAK8Puppi'+postfix+':tau1',
                                                                           'NjettinessAK8Puppi'+postfix+':tau2',
                                                                           'NjettinessAK8Puppi'+postfix+':tau3']

    setattr(process,"ak8PFJetsPuppiValueMap"+postfix, 
            cms.EDProducer("RecoJetToPatJetDeltaRValueMapProducer",
                           src = cms.InputTag("ak8PFJetsCHS"+postfix),
                           matched = cms.InputTag("patJetsAK8Puppi"+postfix),                                         
                           distMax = cms.double(0.8),
                           values = cms.vstring([
                    'userFloat("NjettinessAK8Puppi'+postfix+':tau1")',
                    'userFloat("NjettinessAK8Puppi'+postfix+':tau2")',
                    'userFloat("NjettinessAK8Puppi'+postfix+':tau3")',
                    'pt','eta','phi','mass'
                    ]),
                           valueLabels = cms.vstring( [
                    'NjettinessAK8PuppiTau1',
                    'NjettinessAK8PuppiTau2',
                    'NjettinessAK8PuppiTau3',
                    'pt','eta','phi','mass'
                    ])
                           )
            )
    #process.patJetsAK8.userData.userFloats.src += ['NjettinessAK8:tau1','NjettinessAK8:tau2','NjettinessAK8:tau3']

    getattr(process,"patJetsAK8"+postfix).userData.userFloats.src += [cms.InputTag('ak8PFJetsPuppiValueMap'+postfix,'NjettinessAK8PuppiTau1'),
                                                                      cms.InputTag('ak8PFJetsPuppiValueMap'+postfix,'NjettinessAK8PuppiTau2'),
                                                                      cms.InputTag('ak8PFJetsPuppiValueMap'+postfix,'NjettinessAK8PuppiTau3'),
                                                                      cms.InputTag('ak8PFJetsPuppiValueMap'+postfix,'pt'),
                                                                      cms.InputTag('ak8PFJetsPuppiValueMap'+postfix,'eta'),
                                                                      cms.InputTag('ak8PFJetsPuppiValueMap'+postfix,'phi'),
                                                                      cms.InputTag('ak8PFJetsPuppiValueMap'+postfix,'mass'),
                                                                      ]
    
    
    ## PATify pruned fat jets
    addJetCollection(
        process, postfix=postfix,
        labelName = 'AK8PFCHSSoftDrop',
        jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop'+postfix),
        btagDiscriminators = ['None'],
        jetCorrections = ('AK8PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        getJetMCFlavour = False # jet flavor disabled
    )
    
    ## PATify soft drop subjets
    addJetCollection(
        process, postfix=postfix,
        labelName = 'AK8PFCHSSoftDropSubjets',
        jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop'+postfix,'SubJets'),
        algo = 'ak',  # needed for subjet flavor clustering
        rParam = 0.8, # needed for subjet flavor clustering
        btagDiscriminators = ['pfCombinedSecondaryVertexV2BJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags'],
        jetCorrections = ('AK4PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        explicitJTA = True,  # needed for subjet b tagging
        svClustering = True, # needed for subjet b tagging
        genJetCollection = cms.InputTag('slimmedGenJets'), 
        fatJets=cms.InputTag('ak8PFJetsCHS'+postfix),             # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('ak8PFJetsCHSSoftDrop'+postfix) # needed for subjet flavor clustering
    )
    getattr(process,"selectedPatJetsAK8PFCHSSoftDrop"+postfix).cut = cms.string("pt > 170")
    
    setattr(process,"slimmedJetsAK8PFCHSSoftDropSubjets"+postfix,
            cms.EDProducer("PATJetSlimmer",
                           src = cms.InputTag("selectedPatJetsAK8PFCHSSoftDropSubjets"+postfix),
                           packedPFCandidates = cms.InputTag("packedPFCandidates"+postfix),
                           dropJetVars = cms.string("1"),
                           dropDaughters = cms.string("0"),
                           rekeyDaughters = cms.string("1"),
                           dropTrackRefs = cms.string("1"),
                           dropSpecific = cms.string("1"),
                           dropTagInfos = cms.string("1"),
                           modifyJets = cms.bool(True),
                           mixedDaughters = cms.bool(False),
                           modifierConfig = cms.PSet( modifications = cms.VPSet() )
                           )
            )

    
    ## Establish references between PATified fat jets and subjets using the BoostedJetMerger
    setattr(process,"slimmedJetsAK8PFCHSSoftDropPacked"+postfix, 
            cms.EDProducer("BoostedJetMerger",
                           jetSrc=cms.InputTag("selectedPatJetsAK8PFCHSSoftDrop"+postfix),
                           subjetSrc=cms.InputTag("slimmedJetsAK8PFCHSSoftDropSubjets"+postfix) 
                           )
            )
    
    

    ## PATify pruned fat jets
    addJetCollection(
        process, postfix=postfix,
        labelName = 'AK8PFPuppiSoftDrop',
        jetSource = cms.InputTag('ak8PFJetsPuppiSoftDrop'+postfix),
        btagDiscriminators = ['None'],
        jetCorrections = ('AK8PFPuppi', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        getJetMCFlavour = False # jet flavor disabled
    )
    
    ## PATify soft drop subjets
    addJetCollection(
        process, postfix=postfix,
        labelName = 'AK8PFPuppiSoftDropSubjets',
        jetSource = cms.InputTag('ak8PFJetsPuppiSoftDrop'+postfix,'SubJets'),
        algo = 'ak',  # needed for subjet flavor clustering
        rParam = 0.8, # needed for subjet flavor clustering
        btagDiscriminators = ['pfCombinedSecondaryVertexV2BJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags'],
        jetCorrections = ('AK4PFPuppi', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        explicitJTA = True,  # needed for subjet b tagging
        svClustering = True, # needed for subjet b tagging
        genJetCollection = cms.InputTag('slimmedGenJets'), 
        fatJets=cms.InputTag('ak8PFJetsPuppi'+postfix),             # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('ak8PFJetsPuppiSoftDrop'+postfix) # needed for subjet flavor clustering
    )
    getattr(process,"selectedPatJetsAK8PFPuppiSoftDrop"+postfix).cut = cms.string("pt > 170")
    
    setattr(process,"slimmedJetsAK8PFPuppiSoftDropSubjets"+postfix,
            cms.EDProducer("PATJetSlimmer",
        src = cms.InputTag("selectedPatJetsAK8PFPuppiSoftDropSubjets"+postfix),
        packedPFCandidates = cms.InputTag("packedPFCandidates"+postfix),
        dropJetVars = cms.string("1"),
        dropDaughters = cms.string("0"),
        rekeyDaughters = cms.string("1"),
        dropTrackRefs = cms.string("1"),
        dropSpecific = cms.string("1"),
        dropTagInfos = cms.string("1"),
        modifyJets = cms.bool(True),
        mixedDaughters = cms.bool(False),
        modifierConfig = cms.PSet( modifications = cms.VPSet() )
    ) )

    
    ## Establish references between PATified fat jets and subjets using the BoostedJetMerger
    setattr(process,"slimmedJetsAK8PFPuppiSoftDropPacked"+postfix,
            cms.EDProducer("BoostedJetMerger",
        jetSrc=cms.InputTag("selectedPatJetsAK8PFPuppiSoftDrop"+postfix),
        subjetSrc=cms.InputTag("slimmedJetsAK8PFPuppiSoftDropSubjets"+postfix)
    ) )

    
    setattr(process,"packedPatJetsAK8"+postfix,
            cms.EDProducer("JetSubstructurePacker",
            jetSrc = cms.InputTag("selectedPatJetsAK8"+postfix),
            distMax = cms.double(0.8),
            algoTags = cms.VInputTag(
                # NOTE: For an optimal storage of the AK8 jet daughters, the first subjet collection listed here should be
                #       derived from AK8 jets, i.e., subjets should contain either all or a subset of AK8 constituents.
                #       The PUPPI collection has its own pointers to its own PUPPI constituents. 
                cms.InputTag("slimmedJetsAK8PFCHSSoftDropPacked"+postfix),
                cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked"+postfix)
            ),
            algoLabels = cms.vstring(
                'SoftDrop',
                'SoftDropPuppi'
                ),
            fixDaughters = cms.bool(True),
            packedPFCandidates = cms.InputTag("packedPFCandidates"+postfix), #oldPFCandToPackedOrDiscarded #"packedPFCandidates"+postfix
    ) )


    #if the slimmedJet collection is not here, produce it
    if not hasattr(process, "slimmedJetsAK8"+postfix):
        from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi import slimmedJetsAK8
        setattr(process, "slimmedJetsAK8"+postfix, slimmedJetsAK8.clone(
                src = cms.InputTag("packedPatJetsAK8"+postfix),
                packedPFCandidates = cms.InputTag("packedPFCandidates"), #MM FIXME
                ) )

    # switch off daughter re-keying since it's done in the JetSubstructurePacker (and can't be done afterwards)
    getattr(process,"slimmedJetsAK8"+postfix).rekeyDaughters = "0"


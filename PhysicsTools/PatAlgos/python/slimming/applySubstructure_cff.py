import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def applySubstructure( process, postfix="" ) :

    task = getPatAlgosToolsTask(process)

    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection


    from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets as patJetsDefault

    #add AK8
    addJetCollection(process, postfix=postfix, labelName = 'AK8',
                     jetSource = cms.InputTag('ak8PFJetsCHS'+postfix),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    getattr(process,"patJetsAK8"+postfix).userData.userFloats.src = [] # start with empty list of user floats
    getattr(process,"selectedPatJetsAK8").cut = cms.string("pt > 170")


    ## AK8 groomed masses
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSPruned, ak8PFJetsCHSSoftDrop
    addToProcessAndTask('ak8PFJetsCHSPruned'+postfix, ak8PFJetsCHSPruned.clone(), process, task)
    addToProcessAndTask('ak8PFJetsCHSSoftDrop'+postfix, ak8PFJetsCHSSoftDrop.clone(), process, task)
    from RecoJets.JetProducers.ak8PFJetsCHS_groomingValueMaps_cfi import ak8PFJetsCHSPrunedMass, ak8PFJetsCHSTrimmedMass, ak8PFJetsCHSFilteredMass, ak8PFJetsCHSSoftDropMass
    addToProcessAndTask('ak8PFJetsCHSPrunedMass'+postfix, ak8PFJetsCHSPrunedMass.clone(), process, task)
    addToProcessAndTask('ak8PFJetsCHSTrimmedMass'+postfix, ak8PFJetsCHSTrimmedMass.clone(), process, task)
    addToProcessAndTask('ak8PFJetsCHSFilteredMass'+postfix, ak8PFJetsCHSFilteredMass.clone(), process, task)
    addToProcessAndTask('ak8PFJetsCHSSoftDropMass'+postfix, ak8PFJetsCHSSoftDropMass.clone(), process, task)

    getattr(process,"patJetsAK8").userData.userFloats.src += ['ak8PFJetsCHSPrunedMass'+postfix,'ak8PFJetsCHSSoftDropMass'+postfix]  
    getattr(process,"patJetsAK8").addTagInfos = cms.bool(False)



    # add Njetiness
    process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
    task.add(process.Njettiness)
    addToProcessAndTask('NjettinessAK8'+postfix, process.Njettiness.clone(), process, task)


    getattr(process,"NjettinessAK8").src = cms.InputTag("ak8PFJetsCHS"+postfix)
    getattr(process,"NjettinessAK8").cone = cms.double(0.8)
    getattr(process,"patJetsAK8").userData.userFloats.src += ['NjettinessAK8'+postfix+':tau1','NjettinessAK8'+postfix+':tau2','NjettinessAK8'+postfix+':tau3']




    #add AK8 from PUPPI                                                                                                                                 
    from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJetsPuppi
    from RecoJets.JetProducers.ak8PFJets_cfi import ak8PFJetsPuppi
    addToProcessAndTask('ak4PFJetsPuppi'+postfix,ak4PFJetsPuppi.clone(), process, task)
    addToProcessAndTask('ak8PFJetsPuppi'+postfix,ak8PFJetsPuppi.clone(), process, task)
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsPuppiSoftDrop
    addToProcessAndTask('ak8PFJetsPuppiSoftDrop'+postfix, ak8PFJetsPuppiSoftDrop.clone(), process, task)
    getattr(process,"ak8PFJetsPuppi").doAreaFastjet = True # even for standard ak8PFJets this is overwritten in RecoJets/Configuration/python/RecoPFJets_cff


    #add AK8 from PUPPI
    from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJetsPuppi
    from RecoJets.JetProducers.ak8PFJets_cfi import ak8PFJetsPuppi, ak8PFJetsPuppiSoftDrop, ak8PFJetsPuppiConstituents, ak8PFJetsCHSConstituents
    
    #from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsPuppi, ak8PFJetsPuppiSoftDrop, ak8PFJetsPuppiConstituents, ak8PFJetsCHSConstituents
    addToProcessAndTask('ak4PFJetsPuppi'+postfix,ak4PFJetsPuppi.clone(), process, task)
    addToProcessAndTask('ak8PFJetsPuppiConstituents', ak8PFJetsPuppiConstituents.clone(), process, task )
    addToProcessAndTask('ak8PFJetsCHSConstituents', ak8PFJetsCHSConstituents.clone(), process, task )
    addToProcessAndTask('ak8PFJetsPuppi'+postfix,ak8PFJetsPuppi.clone(), process, task)
    addToProcessAndTask('ak8PFJetsPuppiSoftDrop'+postfix, ak8PFJetsPuppiSoftDrop.clone(), process, task)
    getattr(process,"ak8PFJetsPuppi").doAreaFastjet = True # even for standard ak8PFJets this is overwritten in RecoJets/Configuration/python/RecoPFJets_cff

        
    addJetCollection(process, postfix=postfix, labelName = 'AK8Puppi',
                     jetSource = cms.InputTag('ak8PFJetsPuppi'+postfix),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
                     btagDiscriminators = ([x.value() for x in patJetsDefault.discriminatorSources] + ['pfBoostedDoubleSecondaryVertexAK8BJetTags']),
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src = [] # start with empty list of user floats
    getattr(process,"selectedPatJetsAK8Puppi"+postfix).cut = cms.string("pt > 170")


    from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import j2tParametersVX
    addToProcessAndTask('ak8PFJetsPuppiTracksAssociatorAtVertex'+postfix, cms.EDProducer("JetTracksAssociatorAtVertex",
                                      j2tParametersVX.clone( coneSize = cms.double(0.8) ),
                                      jets = cms.InputTag("ak8PFJetsPuppi") ),
                        process, task)
    addToProcessAndTask('patJetAK8PuppiCharge'+postfix, cms.EDProducer("JetChargeProducer",
                                     src = cms.InputTag("ak8PFJetsPuppiTracksAssociatorAtVertex"),
                                     var = cms.string('Pt'),
                                     exp = cms.double(1.0) ), 
                        process, task)

    ## AK8 groomed masses
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsPuppiSoftDrop
    addToProcessAndTask('ak8PFJetsPuppiSoftDrop'+postfix, ak8PFJetsPuppiSoftDrop.clone(), process, task)
    from RecoJets.JetProducers.ak8PFJetsPuppi_groomingValueMaps_cfi import ak8PFJetsPuppiSoftDropMass
    addToProcessAndTask('ak8PFJetsPuppiSoftDropMass'+postfix, ak8PFJetsPuppiSoftDropMass.clone(), process, task)
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src += ['ak8PFJetsPuppiSoftDropMass'+postfix]
    getattr(process,"patJetsAK8Puppi"+postfix).addTagInfos = cms.bool(False)



    # add Njetiness
    addToProcessAndTask('NjettinessAK8Puppi'+postfix, process.Njettiness.clone(), process, task)
    getattr(process,"NjettinessAK8Puppi"+postfix).src = cms.InputTag("ak8PFJetsPuppi"+postfix)
    getattr(process,"NjettinessAK8Puppi").cone = cms.double(0.8)
    getattr(process,"patJetsAK8Puppi").userData.userFloats.src += ['NjettinessAK8Puppi'+postfix+':tau1','NjettinessAK8Puppi'+postfix+':tau2','NjettinessAK8Puppi'+postfix+':tau3']




    addToProcessAndTask("ak8PFJetsCHSValueMap"+postfix, cms.EDProducer("RecoJetToPatJetDeltaRValueMapProducer",
                                            src = cms.InputTag("ak8PFJetsPuppi"+postfix),
                                            matched = cms.InputTag("patJetsAK8"+postfix),
                                            distMax = cms.double(0.8),
                                            values = cms.vstring([
                                                'userFloat("ak8PFJetsCHSPrunedMass"'+postfix+')',
                                                'userFloat("ak8PFJetsCHSSoftDropMass"'+postfix+')',
                                                'userFloat("NjettinessAK8'+postfix+':tau1")',
                                                'userFloat("NjettinessAK8'+postfix+':tau2")',
                                                'userFloat("NjettinessAK8'+postfix+':tau3")',
                                                'pt','eta','phi','mass'
                                            ]),
                                            valueLabels = cms.vstring( [
                                                'ak8PFJetsCHSPrunedMass',
                                                'ak8PFJetsCHSSoftDropMass',
                                                'NjettinessAK8CHSTau1',
                                                'NjettinessAK8CHSTau2',
                                                'NjettinessAK8CHSTau3',
                                                'pt','eta','phi','mass'
                                            ]) ),
                        process, task)

    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src += [
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'ak8PFJetsCHSPrunedMass'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'ak8PFJetsCHSSoftDropMass'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'NjettinessAK8CHSTau1'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'NjettinessAK8CHSTau2'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'NjettinessAK8CHSTau3'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'pt'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'eta'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'phi'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'mass'),
                                                   ]

    # add Njetiness
    process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
    task.add(process.Njettiness)
    addToProcessAndTask('NjettinessAK8Subjets'+postfix, process.Njettiness.clone(), process, task)
    getattr(process,"NjettinessAK8Subjets"+postfix).src = cms.InputTag("ak8PFJetsPuppiSoftDrop"+postfix, "SubJets")
    getattr(process,"NjettinessAK8Subjets").cone = cms.double(0.8)
    

    
    ## PATify CHS soft drop fat jets
    addJetCollection(
        process,
        postfix=postfix,
        labelName = 'AK8PFCHSSoftDrop',
        jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop'+postfix),
        btagDiscriminators = ['None'],
        jetCorrections = ('AK8PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        getJetMCFlavour = False # jet flavor disabled
    )


    ## PATify puppi soft drop fat jets
    addJetCollection(
        process,
        postfix=postfix,
        labelName = 'AK8PFPuppiSoftDrop',
        jetSource = cms.InputTag('ak8PFJetsPuppiSoftDrop'+postfix),
        btagDiscriminators = ['None'],
        jetCorrections = ('AK8PFPuppi', ['L2Relative', 'L3Absolute'], 'None'),
        getJetMCFlavour = False # jet flavor disabled
    )
    
    ## PATify soft drop subjets
    addJetCollection(
        process,
        postfix=postfix,
        labelName = 'AK8PFPuppiSoftDropSubjets',
        jetSource = cms.InputTag('ak8PFJetsPuppiSoftDrop'+postfix,'SubJets'),
        algo = 'ak',  # needed for subjet flavor clustering
        rParam = 0.8, # needed for subjet flavor clustering
        btagDiscriminators = ['pfDeepCSVJetTags:probb', 'pfDeepCSVJetTags:probbb', 'pfCombinedInclusiveSecondaryVertexV2BJetTags','pfCombinedMVAV2BJetTags'],
        jetCorrections = ('AK4PFPuppi', ['L2Relative', 'L3Absolute'], 'None'),
        explicitJTA = True,  # needed for subjet b tagging
        svClustering = True, # needed for subjet b tagging
        genJetCollection = cms.InputTag('slimmedGenJets'), 
        fatJets=cms.InputTag('ak8PFJetsPuppi'),             # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('ak8PFJetsPuppiSoftDrop') # needed for subjet flavor clustering
    )
    getattr(process,"selectedPatJetsAK8PFPuppiSoftDrop"+postfix).cut = cms.string("pt > 170")
    getattr(process,"patJetsAK8PFPuppiSoftDropSubjets"+postfix).userData.userFloats.src += ['NjettinessAK8Subjets'+postfix+':tau1','NjettinessAK8Subjets'+postfix+':tau2','NjettinessAK8Subjets'+postfix+':tau3']
    
    addToProcessAndTask("slimmedJetsAK8PFPuppiSoftDropSubjets"+postfix,
                        cms.EDProducer("PATJetSlimmer",
                             src = cms.InputTag("selectedPatJetsAK8PFPuppiSoftDropSubjets"),
                             packedPFCandidates = cms.InputTag("packedPFCandidates"),
                             dropJetVars = cms.string("1"),
                             dropDaughters = cms.string("0"),
                             rekeyDaughters = cms.string("1"),
                             dropTrackRefs = cms.string("1"),
                             dropSpecific = cms.string("1"),
                             dropTagInfos = cms.string("1"),
                             modifyJets = cms.bool(True),
                             mixedDaughters = cms.bool(False),
                             modifierConfig = cms.PSet( modifications = cms.VPSet() )
                                       ),
                        process, task)

    
    ## Establish references between PATified fat jets and subjets using the BoostedJetMerger
    addToProcessAndTask("slimmedJetsAK8PFPuppiSoftDropPacked"+postfix,
                        cms.EDProducer("BoostedJetMerger",
                               jetSrc=cms.InputTag("selectedPatJetsAK8PFPuppiSoftDrop"),
                               subjetSrc=cms.InputTag("slimmedJetsAK8PFPuppiSoftDropSubjets")
                                       ),
                        process, task )

    
    addToProcessAndTask("packedPatJetsAK8"+postfix, cms.EDProducer("JetSubstructurePacker",
                                           jetSrc = cms.InputTag("selectedPatJetsAK8Puppi"+postfix),
                                           distMax = cms.double(0.8),
                                           algoTags = cms.VInputTag(
                                               cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked"+postfix)
                                           ),
                                           algoLabels = cms.vstring(
                                               'SoftDropPuppi'
                                           ),
                                          fixDaughters = cms.bool(True),
                                          packedPFCandidates = cms.InputTag("packedPFCandidates"+postfix),
                                                                   ),
                        process, task)

    # switch off daughter re-keying since it's done in the JetSubstructurePacker (and can't be done afterwards)
    process.slimmedJetsAK8.rekeyDaughters = "0"


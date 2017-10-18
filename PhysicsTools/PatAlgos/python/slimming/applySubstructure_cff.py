import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def applySubstructure( process, postfix="" ) :

    task = getPatAlgosToolsTask(process)

    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection


    from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets as patJetsDefault


    # Configure the RECO jets
    from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJetsPuppi
    from RecoJets.JetProducers.ak8PFJets_cfi import ak8PFJetsPuppi, ak8PFJetsPuppiSoftDrop, ak8PFJetsPuppiConstituents, ak8PFJetsCHSConstituents
    from RecoJets.JetProducers.ak8GenJets_cfi import ak8GenJets, ak8GenJetsSoftDrop, ak8GenJetsConstituents
    addToProcessAndTask('ak4PFJetsPuppi'+postfix,ak4PFJetsPuppi.clone(), process, task)
    addToProcessAndTask('ak8PFJetsPuppi'+postfix,ak8PFJetsPuppi.clone(), process, task)
    addToProcessAndTask('ak8PFJetsPuppiConstituents', ak8PFJetsPuppiConstituents.clone(cut = cms.string('pt > 170.0 && abs(rapidity()) < 2.4') ), process, task )
    addToProcessAndTask('ak8PFJetsCHSConstituents', ak8PFJetsCHSConstituents.clone(), process, task )
    addToProcessAndTask('ak8PFJetsPuppiSoftDrop'+postfix, ak8PFJetsPuppiSoftDrop.clone( src = cms.InputTag('ak8PFJetsPuppiConstituents', 'constituents') ), process, task)
    addToProcessAndTask('ak8GenJetsNoNuConstituents'+postfix, ak8GenJetsConstituents.clone(src='ak8GenJetsNoNu'), process, task )
    addToProcessAndTask('ak8GenJetsNoNuSoftDrop'+postfix,ak8GenJetsSoftDrop.clone(src=cms.InputTag('ak8GenJetsNoNuConstituents'+postfix, 'constituents')),process,task)
    addToProcessAndTask('slimmedGenJetsAK8SoftDropSubJets'+postfix,
                            cms.EDProducer("PATGenJetSlimmer",
                                               src = cms.InputTag("ak8GenJetsNoNuSoftDrop"+postfix, "SubJets"),
                                               packedGenParticles = cms.InputTag("packedGenParticles"),
                                               cut = cms.string(""),
                                               cutLoose = cms.string(""),
                                               nLoose = cms.uint32(0),
                                               clearDaughters = cms.bool(False), #False means rekeying
                                               dropSpecific = cms.bool(True),  # Save space
                                               ), process, task )
    
    
    #add AK8 CHS
    addJetCollection(process, postfix=postfix, labelName = 'AK8',
                     jetSource = cms.InputTag('ak8PFJetsCHS'+postfix),
                     algo= 'AK', rParam = 0.8,
                     btagDiscriminators = ['None'],
                     jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    getattr(process,"patJetsAK8"+postfix).userData.userFloats.src = [] # start with empty list of user floats
    getattr(process,"selectedPatJetsAK8").cut = cms.string("pt > 170")


    ## add AK8 groomed masses with CHS
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

    # add Njetiness for CHS
    process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
    task.add(process.Njettiness)
    addToProcessAndTask('NjettinessAK8'+postfix, process.Njettiness.clone(), process, task)
    getattr(process,"NjettinessAK8").src = cms.InputTag("ak8PFJetsCHS"+postfix)
    getattr(process,"NjettinessAK8").cone = cms.double(0.8)
    getattr(process,"patJetsAK8").userData.userFloats.src += ['NjettinessAK8'+postfix+':tau1','NjettinessAK8'+postfix+':tau2','NjettinessAK8'+postfix+':tau3','NjettinessAK8'+postfix+':tau4']

    # add Njetiness from CHS
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





    #add RECO AK8 from PUPPI and RECO AK8 PUPPI with soft drop... will be needed by ungroomed AK8 jets later
    ## PATify puppi soft drop fat jets
    addJetCollection(
        process,
        postfix=postfix,
        labelName = 'AK8PFPuppiSoftDrop' + postfix,
        jetSource = cms.InputTag('ak8PFJetsPuppiSoftDrop'+postfix),
        btagDiscriminators = ['None'],
        genJetCollection = cms.InputTag('slimmedGenJetsAK8'), 
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
        genJetCollection = cms.InputTag('slimmedGenJetsAK8SoftDropSubJets'), 
        fatJets=cms.InputTag('ak8PFJetsPuppi'),             # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('ak8PFJetsPuppiSoftDrop') # needed for subjet flavor clustering
    )


    # add groomed ECFs and N-subjettiness to soft dropped pat::Jets for fat jets and subjets
    process.load('RecoJets.JetProducers.ECF_cff')
    addToProcessAndTask('nb1AK8PuppiSoftDrop'+postfix, process.ecfNbeta1.clone(src = cms.InputTag("ak8PFJetsPuppiSoftDrop"+postfix), cuts = cms.vstring('', '', 'pt > 250')), process, task)
    addToProcessAndTask('nb2AK8PuppiSoftDrop'+postfix, process.ecfNbeta2.clone(src = cms.InputTag("ak8PFJetsPuppiSoftDrop"+postfix), cuts = cms.vstring('', '', 'pt > 250')), process, task)

    #too slow now ==> disable
    from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
    pp_on_XeXe_2017.toModify(getattr(process,'nb1AK8PuppiSoftDrop'+postfix), cuts = ['pt > 999999', 'pt > 999999', 'pt > 999999'] )
    pp_on_XeXe_2017.toModify(getattr(process,'nb2AK8PuppiSoftDrop'+postfix), cuts = ['pt > 999999', 'pt > 999999', 'pt > 999999'] )
    from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
    phase2_common.toModify(getattr(process,'nb1AK8PuppiSoftDrop'+postfix), cuts = ['pt > 999999', 'pt > 999999', 'pt > 999999'] )
    phase2_common.toModify(getattr(process,'nb2AK8PuppiSoftDrop'+postfix), cuts = ['pt > 999999', 'pt > 999999', 'pt > 999999'] )

    getattr(process,"patJetsAK8PFPuppiSoftDrop").userData.userFloats.src += ['nb1AK8PuppiSoftDrop'+postfix+':ecfN2','nb1AK8PuppiSoftDrop'+postfix+':ecfN3']
    getattr(process,"patJetsAK8PFPuppiSoftDrop").userData.userFloats.src += ['nb2AK8PuppiSoftDrop'+postfix+':ecfN2','nb2AK8PuppiSoftDrop'+postfix+':ecfN3']
    addToProcessAndTask('nb1AK8PuppiSoftDropSubjets'+postfix, process.ecfNbeta1.clone(src = cms.InputTag("ak8PFJetsPuppiSoftDrop"+postfix, "SubJets")), process, task)
    addToProcessAndTask('nb2AK8PuppiSoftDropSubjets'+postfix, process.ecfNbeta2.clone(src = cms.InputTag("ak8PFJetsPuppiSoftDrop"+postfix, "SubJets")), process, task)
    getattr(process,"patJetsAK8PFPuppiSoftDropSubjets"+postfix).userData.userFloats.src += ['nb1AK8PuppiSoftDropSubjets'+postfix+':ecfN2','nb1AK8PuppiSoftDropSubjets'+postfix+':ecfN3']
    getattr(process,"patJetsAK8PFPuppiSoftDropSubjets"+postfix).userData.userFloats.src += ['nb2AK8PuppiSoftDropSubjets'+postfix+':ecfN2','nb2AK8PuppiSoftDropSubjets'+postfix+':ecfN3']
    getattr(process,"patJetsAK8PFPuppiSoftDropSubjets"+postfix).userData.userFloats.src += ['NjettinessAK8Subjets'+postfix+':tau1','NjettinessAK8Subjets'+postfix+':tau2','NjettinessAK8Subjets'+postfix+':tau3','NjettinessAK8Subjets'+postfix+':tau4']

    pp_on_XeXe_2017.toModify(getattr(process,'nb1AK8PuppiSoftDropSubjets'+postfix), cuts = ['pt > 999999', 'pt > 999999', 'pt > 999999'] )
    pp_on_XeXe_2017.toModify(getattr(process,'nb2AK8PuppiSoftDropSubjets'+postfix), cuts = ['pt > 999999', 'pt > 999999', 'pt > 999999'] )
    phase2_common.toModify(getattr(process,'nb1AK8PuppiSoftDropSubjets'+postfix), cuts = ['pt > 999999', 'pt > 999999', 'pt > 999999'] )
    phase2_common.toModify(getattr(process,'nb2AK8PuppiSoftDropSubjets'+postfix), cuts = ['pt > 999999', 'pt > 999999', 'pt > 999999'] )

    # rekey the groomed ECF value maps to the ungroomed reco jets, which will then be picked
    # up by PAT in the user floats. 
    addToProcessAndTask("ak8PFJetsPuppiSoftDropValueMap"+postfix, 
                        cms.EDProducer("RecoJetToPatJetDeltaRValueMapProducer",
                                       src = cms.InputTag("ak8PFJetsPuppi"+postfix),
                                       matched = cms.InputTag("patJetsAK8PFPuppiSoftDrop"+postfix),
                                       distMax = cms.double(0.8),
                                       values = cms.vstring([
                    'userFloat("nb1AK8PuppiSoftDrop'+postfix+':ecfN2")',
                    'userFloat("nb1AK8PuppiSoftDrop'+postfix+':ecfN3")',
                    'userFloat("nb2AK8PuppiSoftDrop'+postfix+':ecfN2")',
                    'userFloat("nb2AK8PuppiSoftDrop'+postfix+':ecfN3")',
                    ]),
                                       valueLabels = cms.vstring( [
                    'nb1AK8PuppiSoftDropN2',
                    'nb1AK8PuppiSoftDropN3',
                    'nb2AK8PuppiSoftDropN2',
                    'nb2AK8PuppiSoftDropN3',
                    ]) ),
                    process, task)

        
    # Patify AK8 PF PUPPI
    addJetCollection(process, postfix=postfix, labelName = 'AK8Puppi',
                     jetSource = cms.InputTag('ak8PFJetsPuppi'+postfix),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
                     btagDiscriminators = ([
                         'pfCombinedSecondaryVertexV2BJetTags',
                         'pfCombinedInclusiveSecondaryVertexV2BJetTags',
                         'pfCombinedMVAV2BJetTags',
                         'pfDeepCSVJetTags:probb',
                         'pfDeepCSVJetTags:probc',
                         'pfDeepCSVJetTags:probudsg',
                         'pfDeepCSVJetTags:probbb',
                         'pfBoostedDoubleSecondaryVertexAK8BJetTags']),
                     genJetCollection = cms.InputTag('slimmedGenJetsAK8')
                     )
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src = [] # start with empty list of user floats
    getattr(process,"selectedPatJetsAK8Puppi"+postfix).cut = cms.string("pt > 100")
    getattr(process,"selectedPatJetsAK8Puppi"+postfix).cutLoose = cms.string("pt > 30")
    getattr(process,"selectedPatJetsAK8Puppi"+postfix).nLoose = cms.uint32(3)

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

    ## now add AK8 groomed masses and ECF
    from RecoJets.JetProducers.ak8PFJetsPuppi_groomingValueMaps_cfi import ak8PFJetsPuppiSoftDropMass
    addToProcessAndTask('ak8PFJetsPuppiSoftDropMass'+postfix, ak8PFJetsPuppiSoftDropMass.clone(), process, task)
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src += ['ak8PFJetsPuppiSoftDropMass'+postfix]
    getattr(process,"patJetsAK8Puppi"+postfix).addTagInfos = cms.bool(False)
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src += [
        cms.InputTag('ak8PFJetsPuppiSoftDropValueMap'+postfix,'nb1AK8PuppiSoftDropN2'),
        cms.InputTag('ak8PFJetsPuppiSoftDropValueMap'+postfix,'nb1AK8PuppiSoftDropN3'),
        cms.InputTag('ak8PFJetsPuppiSoftDropValueMap'+postfix,'nb2AK8PuppiSoftDropN2'),
        cms.InputTag('ak8PFJetsPuppiSoftDropValueMap'+postfix,'nb2AK8PuppiSoftDropN3'),
        ]


    # add PUPPI Njetiness
    addToProcessAndTask('NjettinessAK8Puppi'+postfix, process.Njettiness.clone(), process, task)
    getattr(process,"NjettinessAK8Puppi"+postfix).src = cms.InputTag("ak8PFJetsPuppi"+postfix)
    getattr(process,"NjettinessAK8Puppi").cone = cms.double(0.8)
    getattr(process,"patJetsAK8Puppi").userData.userFloats.src += ['NjettinessAK8Puppi'+postfix+':tau1','NjettinessAK8Puppi'+postfix+':tau2','NjettinessAK8Puppi'+postfix+':tau3','NjettinessAK8Puppi'+postfix+':tau4']

    # Now combine the CHS and PUPPI information into the PUPPI jets via delta R value maps
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
                                                'userFloat("NjettinessAK8'+postfix+':tau4")',
                                                'pt','eta','phi','mass'
                                            ]),
                                            valueLabels = cms.vstring( [
                                                'ak8PFJetsCHSPrunedMass',
                                                'ak8PFJetsCHSSoftDropMass',
                                                'NjettinessAK8CHSTau1',
                                                'NjettinessAK8CHSTau2',
                                                'NjettinessAK8CHSTau3',
                                                'NjettinessAK8CHSTau4',
                                                'pt','eta','phi','mass'
                                            ]) ),
                        process, task)


    # Now set up the user floats
    getattr(process,"patJetsAK8Puppi"+postfix).userData.userFloats.src += [
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'ak8PFJetsCHSPrunedMass'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'ak8PFJetsCHSSoftDropMass'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'NjettinessAK8CHSTau1'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'NjettinessAK8CHSTau2'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'NjettinessAK8CHSTau3'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'NjettinessAK8CHSTau4'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'pt'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'eta'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'phi'),
                                                   cms.InputTag('ak8PFJetsCHSValueMap'+postfix,'mass'),
                                                   ]

    

    
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
    # Reconfigure the slimmedAK8 jet information to keep 
    process.slimmedJetsAK8.dropDaughters = cms.string("pt < 170")
    process.slimmedJetsAK8.dropSpecific = cms.string("pt < 170")
    process.slimmedJetsAK8.dropTagInfos = cms.string("pt < 170")

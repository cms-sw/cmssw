import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *

def miniAOD_customizeCommon(process):
    process.patMuons.isoDeposits = cms.PSet()
    process.patElectrons.isoDeposits = cms.PSet()
    process.patTaus.isoDeposits = cms.PSet()
    process.patPhotons.isoDeposits = cms.PSet()
    #
    process.patMuons.embedTrack         = True  # used for IDs
    process.patMuons.embedCombinedMuon  = True  # used for IDs
    process.patMuons.embedMuonBestTrack = True  # used for IDs
    process.patMuons.embedStandAloneMuon = True # maybe?
    process.patMuons.embedPickyMuon = False   # no, use best track
    process.patMuons.embedTpfmsMuon = False   # no, use best track
    process.patMuons.embedDytMuon   = False   # no, use best track
    #
    # disable embedding of electron and photon associated objects already stored by the ReducedEGProducer
    process.patElectrons.embedGsfElectronCore = False  ## process.patElectrons.embed in AOD externally stored gsf electron core
    process.patElectrons.embedSuperCluster    = False  ## process.patElectrons.embed in AOD externally stored supercluster
    process.patElectrons.embedPflowSuperCluster         = False  ## process.patElectrons.embed in AOD externally stored supercluster
    process.patElectrons.embedSeedCluster               = False  ## process.patElectrons.embed in AOD externally stored the electron's seedcluster
    process.patElectrons.embedBasicClusters             = False  ## process.patElectrons.embed in AOD externally stored the electron's basic clusters
    process.patElectrons.embedPreshowerClusters         = False  ## process.patElectrons.embed in AOD externally stored the electron's preshower clusters
    process.patElectrons.embedPflowBasicClusters        = False  ## process.patElectrons.embed in AOD externally stored the electron's pflow basic clusters
    process.patElectrons.embedPflowPreshowerClusters    = False  ## process.patElectrons.embed in AOD externally stored the electron's pflow preshower clusters
    process.patElectrons.embedRecHits         = False  ## process.patElectrons.embed in AOD externally stored the RecHits - can be called from the PATElectronProducer
    process.patElectrons.electronSource = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.patElectrons.electronIDSources = cms.PSet(
            # configure many IDs as InputTag <someName> = <someTag> you
            # can comment out those you don't want to save some disk space
            eidRobustLoose      = cms.InputTag("reducedEgamma","eidRobustLoose"),
            eidRobustTight      = cms.InputTag("reducedEgamma","eidRobustTight"),
            eidLoose            = cms.InputTag("reducedEgamma","eidLoose"),
            eidTight            = cms.InputTag("reducedEgamma","eidTight"),
            eidRobustHighEnergy = cms.InputTag("reducedEgamma","eidRobustHighEnergy"),
        )
    process.elPFIsoDepositChargedPAT.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.elPFIsoDepositChargedAllPAT.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.elPFIsoDepositNeutralPAT.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.elPFIsoDepositGammaPAT.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.elPFIsoDepositPUPAT.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    #
    process.patPhotons.embedSuperCluster = False ## whether to process.patPhotons.embed in AOD externally stored supercluster
    process.patPhotons.embedSeedCluster               = False  ## process.patPhotons.embed in AOD externally stored the photon's seedcluster
    process.patPhotons.embedBasicClusters             = False  ## process.patPhotons.embed in AOD externally stored the photon's basic clusters
    process.patPhotons.embedPreshowerClusters         = False  ## process.patPhotons.embed in AOD externally stored the photon's preshower clusters
    process.patPhotons.embedRecHits         = False  ## process.patPhotons.embed in AOD externally stored the RecHits - can be called from the PATPhotonProducer
    process.patPhotons.photonSource = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.patPhotons.photonIDSources = cms.PSet(
                PhotonCutBasedIDLoose = cms.InputTag('reducedEgamma',
                                                      'PhotonCutBasedIDLoose'),
                PhotonCutBasedIDTight = cms.InputTag('reducedEgamma',
                                                      'PhotonCutBasedIDTight')
              )

    process.phPFIsoDepositChargedPAT.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.phPFIsoDepositChargedAllPAT.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.phPFIsoDepositNeutralPAT.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.phPFIsoDepositGammaPAT.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.phPFIsoDepositPUPAT.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    #
    process.selectedPatJets.cut = cms.string("pt > 10")
    process.selectedPatMuons.cut = cms.string("pt > 5 || isPFMuon || (pt > 3 && (isGlobalMuon || isStandAloneMuon || numberOfMatches > 0 || muonID('RPCMuLoose')))")
    process.selectedPatElectrons.cut = cms.string("")
    process.selectedPatTaus.cut = cms.string("pt > 18. && tauID('decayModeFinding')> 0.5")
    process.selectedPatPhotons.cut = cms.string("")

    # add CMS top tagger
    from RecoJets.JetProducers.caTopTaggers_cff import caTopTagInfos
    process.caTopTagInfos = caTopTagInfos.clone()
    process.caTopTagInfosPAT = cms.EDProducer("RecoJetDeltaRTagInfoValueMapProducer",
                                    src = cms.InputTag("ak8PFJetsCHS"),
                                    matched = cms.InputTag("cmsTopTagPFJetsCHS"),
                                    matchedTagInfos = cms.InputTag("caTopTagInfos"),
                                    distMax = cms.double(0.8)
                        )

    ## Add AK8
    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    addJetCollection(process, labelName = 'AK8',
                     jetSource = cms.InputTag('ak8PFJetsCHS'),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     genJetCollection = cms.InputTag('ak8GenJets'),
                     btagInfos = ['caTopTagInfosPAT']
                     )
    process.patJetsAK8.userData.userFloats.src = [] # start with empty list of user floats
    process.selectedPatJetsAK8.cut = cms.string("pt > 100")
    ## Add AK8 soft drop jets
    addJetCollection(
        process,
        labelName = 'AK8PFCHSSoftDrop',
        jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop'),
        btagDiscriminators = ['None'], # turn-off b tagging
        jetCorrections = ('AK8PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        genJetCollection = cms.InputTag('ak8GenJets'),
        getJetMCFlavour = False # jet flavor needs to be disabled for groomed fat jets
    )
    ## Add AK8 soft drop subjets
    addJetCollection(
        process,
        labelName = 'AK8PFCHSSoftDropSubjets',
        jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop','SubJets'),
        algo = 'AK',  # needed for subjet flavor clustering
        rParam = 0.8, # needed for subjet flavor clustering
        btagDiscriminators = [x.getModuleLabel() for x in process.patJets.discriminatorSources], # Use the same b-tag discriminators as for ak4 jets
        jetCorrections = ('AK4PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'), # Using AK4 JECs for subjets which is not entirely appropriate
        genJetCollection = cms.InputTag('ak4GenJets'), # Using ak4GenJets for matching which is not entirely appropriate
        explicitJTA = True,  # needed for subjet b tagging
        svClustering = True, # needed for subjet b tagging
        fatJets=cms.InputTag('ak8PFJetsCHS'),               # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('ak8PFJetsCHSSoftDrop') # needed for subjet flavor clustering
    )
    ## Re-establish references between PATified AK8 soft drop jets and subjets using the BoostedJetMerger
    process.selectedPatJetsAK8PFCHSSoftDropPacked = cms.EDProducer("BoostedJetMerger",
        jetSrc=cms.InputTag("selectedPatJetsAK8PFCHSSoftDrop"),
        subjetSrc=cms.InputTag("selectedPatJetsAK8PFCHSSoftDropSubjets")
    )
    ## Add CMS top tagger jets
    addJetCollection(
        process,
        labelName = 'CMSTopTagPFJetsCHS',
        jetSource = cms.InputTag('cmsTopTagPFJetsCHS'),
        btagDiscriminators = ['None'], # turn-off b tagging
        jetCorrections = ('AK8PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        genJetCollection = cms.InputTag('ak8GenJets'),
        getJetMCFlavour = False # jet flavor needs to be disabled for groomed fat jets
    )
    ## Add CMS top tagger subjets
    from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSConstituents, ca8PFJetsCHS
    process.ak8PFJetsCHSConstituents = ak8PFJetsCHSConstituents.clone() # needed for subjet flavor clustering
    process.ca8PFJetsCHS = ca8PFJetsCHS.clone(src = cms.InputTag("ak8PFJetsCHSConstituents", "constituents"), doAreaFastjet = cms.bool(False)) # needed for subjet flavor clustering
    addJetCollection(
        process,
        labelName = 'CMSTopTagPFJetsCHSSubjets',
        jetSource = cms.InputTag('cmsTopTagPFJetsCHS','caTopSubJets'),
        algo = 'CA',  # needed for subjet flavor clustering
        rParam = 0.8, # needed for subjet flavor clustering
        btagDiscriminators = [x.getModuleLabel() for x in process.patJets.discriminatorSources], # Use the same b-tag discriminators as for ak4 jets
        jetCorrections = ('AK4PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'), # Using AK4 JECs for subjets which is not entirely appropriate
        genJetCollection = cms.InputTag('ak4GenJets'), # Using ak4GenJets for matching which is not entirely appropriate
        explicitJTA = True,  # needed for subjet b tagging
        svClustering = True, # needed for subjet b tagging
        fatJets=cms.InputTag('ca8PFJetsCHS'),             # needed for subjet flavor clustering
        groomedFatJets=cms.InputTag('cmsTopTagPFJetsCHS') # needed for subjet flavor clustering
    )
    ## Re-establish references between PATified CMS top tagger jets and subjets using the BoostedJetMerger
    process.selectedPatJetsCMSTopTagPFJetsCHSPacked = cms.EDProducer("BoostedJetMerger",
        jetSrc=cms.InputTag("selectedPatJetsCMSTopTagPFJetsCHS"),
        subjetSrc=cms.InputTag("selectedPatJetsCMSTopTagPFJetsCHSSubjets")
    )
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

    # Add Njettiness
    process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
    process.NjettinessAK8 = process.Njettiness.clone()
    process.NjettinessAK8.src = cms.InputTag("ak8PFJetsCHS")
    process.NjettinessAK8.cone = cms.double(0.8)
    process.patJetsAK8.userData.userFloats.src += ['NjettinessAK8:tau1','NjettinessAK8:tau2','NjettinessAK8:tau3']

    ## Add soft drop subjet info
    process.ak8PFJetsCHSSDSubJet = cms.EDProducer("RecoJetDeltaRValueMapProducer",
        src = cms.InputTag("ak8PFJetsCHS"),
        matched = cms.InputTag("selectedPatJetsAK8PFCHSSoftDropPacked"),
        distMax = cms.double(0.8),
        values = cms.vstring(
            "? numberOfDaughters > 0 ? daughterPtr(0).pt : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).pt : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).eta : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).eta : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).phi : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).phi : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).mass : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).mass : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).jecFactor(0) : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).jecFactor(0) : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).partonFlavour : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).partonFlavour : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags') : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags') : 0"
        ),
        valueLabels = cms.vstring(
            "0Pt",
            "1Pt",
            "0Eta",
            "1Eta",
            "0Phi",
            "1Phi",
            "0Mass",
            "1Mass",
            "0JecFactor0",
            "1JecFactor0",
            "0Flavour",
            "1Flavour",
            "0CSVv2IVF",
            "1CSVv2IVF"
        ),
        lazyParser = cms.bool(True)
    )
    process.patJetsAK8.userData.userFloats.src += [ 'ak8PFJetsCHSSDSubJet:0Pt'        , 'ak8PFJetsCHSSDSubJet:1Pt'
                                                   ,'ak8PFJetsCHSSDSubJet:0Eta'       , 'ak8PFJetsCHSSDSubJet:1Eta'
                                                   ,'ak8PFJetsCHSSDSubJet:0Phi'       , 'ak8PFJetsCHSSDSubJet:1Phi'
                                                   ,'ak8PFJetsCHSSDSubJet:0Mass'      , 'ak8PFJetsCHSSDSubJet:1Mass'
                                                   ,'ak8PFJetsCHSSDSubJet:0JecFactor0', 'ak8PFJetsCHSSDSubJet:1JecFactor0'
                                                   ,'ak8PFJetsCHSSDSubJet:0Flavour'   , 'ak8PFJetsCHSSDSubJet:1Flavour'
                                                   ,'ak8PFJetsCHSSDSubJet:0CSVv2IVF'  , 'ak8PFJetsCHSSDSubJet:1CSVv2IVF']

    ## Add CMS top tagger subjet info
    process.ak8PFJetsCHSTopSubJet = cms.EDProducer("RecoJetDeltaRValueMapProducer",
        src = cms.InputTag("ak8PFJetsCHS"),
        matched = cms.InputTag("selectedPatJetsCMSTopTagPFJetsCHSPacked"),
        distMax = cms.double(0.8),
        values = cms.vstring(
            "? numberOfDaughters > 0 ? daughterPtr(0).pt : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).pt : 0",
            "? numberOfDaughters > 2 ? daughterPtr(2).pt : 0",
            "? numberOfDaughters > 3 ? daughterPtr(3).pt : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).eta : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).eta : 0",
            "? numberOfDaughters > 2 ? daughterPtr(2).eta : 0",
            "? numberOfDaughters > 3 ? daughterPtr(3).eta : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).phi : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).phi : 0",
            "? numberOfDaughters > 2 ? daughterPtr(2).phi : 0",
            "? numberOfDaughters > 3 ? daughterPtr(3).phi : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).mass : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).mass : 0",
            "? numberOfDaughters > 2 ? daughterPtr(2).mass : 0",
            "? numberOfDaughters > 3 ? daughterPtr(3).mass : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).jecFactor(0) : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).jecFactor(0) : 0",
            "? numberOfDaughters > 2 ? daughterPtr(2).jecFactor(0) : 0",
            "? numberOfDaughters > 3 ? daughterPtr(3).jecFactor(0) : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).partonFlavour : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).partonFlavour : 0",
            "? numberOfDaughters > 2 ? daughterPtr(2).partonFlavour : 0",
            "? numberOfDaughters > 3 ? daughterPtr(3).partonFlavour : 0",
            "? numberOfDaughters > 0 ? daughterPtr(0).bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags') : 0",
            "? numberOfDaughters > 1 ? daughterPtr(1).bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags') : 0",
            "? numberOfDaughters > 2 ? daughterPtr(2).bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags') : 0",
            "? numberOfDaughters > 3 ? daughterPtr(3).bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags') : 0",
        ),
        valueLabels = cms.vstring(
            "0Pt",
            "1Pt",
            "2Pt",
            "3Pt",
            "0Eta",
            "1Eta",
            "2Eta",
            "3Eta",
            "0Phi",
            "1Phi",
            "2Phi",
            "3Phi",
            "0Mass",
            "1Mass",
            "2Mass",
            "3Mass",
            "0JecFactor0",
            "1JecFactor0",
            "2JecFactor0",
            "3JecFactor0",
            "0Flavour",
            "1Flavour",
            "2Flavour",
            "3Flavour",
            "0CSVv2IVF",
            "1CSVv2IVF",
            "2CSVv2IVF",
            "3CSVv2IVF"
        ),
        lazyParser = cms.bool(True)
    )
    process.patJetsAK8.userData.userFloats.src += [ 'ak8PFJetsCHSTopSubJet:0Pt'        , 'ak8PFJetsCHSTopSubJet:1Pt'        , 'ak8PFJetsCHSTopSubJet:2Pt'        , 'ak8PFJetsCHSTopSubJet:3Pt'
                                                   ,'ak8PFJetsCHSTopSubJet:0Eta'       , 'ak8PFJetsCHSTopSubJet:1Eta'       , 'ak8PFJetsCHSTopSubJet:2Eta'       , 'ak8PFJetsCHSTopSubJet:3Eta'
                                                   ,'ak8PFJetsCHSTopSubJet:0Phi'       , 'ak8PFJetsCHSTopSubJet:1Phi'       , 'ak8PFJetsCHSTopSubJet:2Phi'       , 'ak8PFJetsCHSTopSubJet:3Phi'
                                                   ,'ak8PFJetsCHSTopSubJet:0Mass'      , 'ak8PFJetsCHSTopSubJet:1Mass'      , 'ak8PFJetsCHSTopSubJet:2Mass'      , 'ak8PFJetsCHSTopSubJet:3Mass'
                                                   ,'ak8PFJetsCHSTopSubJet:0JecFactor0', 'ak8PFJetsCHSTopSubJet:1JecFactor0', 'ak8PFJetsCHSTopSubJet:2JecFactor0', 'ak8PFJetsCHSTopSubJet:3JecFactor0'
                                                   ,'ak8PFJetsCHSTopSubJet:0Flavour'   , 'ak8PFJetsCHSTopSubJet:1Flavour'   , 'ak8PFJetsCHSTopSubJet:2Flavour'   , 'ak8PFJetsCHSTopSubJet:3Flavour'
                                                   ,'ak8PFJetsCHSTopSubJet:0CSVv2IVF'  , 'ak8PFJetsCHSTopSubJet:1CSVv2IVF'  , 'ak8PFJetsCHSTopSubJet:2CSVv2IVF'  , 'ak8PFJetsCHSTopSubJet:3CSVv2IVF']

    #
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTriggerStandAlone
    switchOnTriggerStandAlone( process, outputModule = '' )
    process.patTrigger.packTriggerPathNames = cms.bool(True)
    #
    # apply type I/type I + II PFMEt corrections to pat::MET object
    # and estimate systematic uncertainties on MET
    # FIXME: this and the typeI MET should become AK4 once we have the proper JEC?
    from PhysicsTools.PatUtils.tools.runType1PFMEtUncertainties import runType1PFMEtUncertainties
    addJetCollection(process, postfix   = "ForMetUnc", labelName = 'AK4PF', jetSource = cms.InputTag('ak4PFJets'), jetCorrections = ('AK4PF', ['L1FastJet', 'L2Relative', 'L3Absolute'], ''))
    process.patJetsAK4PFForMetUnc.getJetMCFlavour = False
    runType1PFMEtUncertainties(process,
                               addToPatDefaultSequence=False,
                               jetCollection="selectedPatJetsAK4PFForMetUnc",
                               electronCollection="selectedPatElectrons",
                               muonCollection="selectedPatMuons",
                               tauCollection="selectedPatTaus",
                               makeType1p2corrPFMEt=True,
                               outputModule=None)


    #keep this after all addJetCollections otherwise it will attempt computing them also for stuf with no taginfos
    #Some useful BTAG vars
    process.patJets.userData.userFunctions = cms.vstring(
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).p4.M):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).nTracks):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().flightDistance(0).value):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().flightDistance(0).significance):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).p4.x):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).p4.y):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).p4.z):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).position.x):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).position.y):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).position.z):(0)',
    )
    process.patJets.userData.userFunctionLabels = cms.vstring('vtxMass','vtxNtracks','vtx3DVal','vtx3DSig','vtxPx','vtxPy','vtxPz','vtxPosX','vtxPosY','vtxPosZ')
    process.patJets.tagInfoSources = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
    process.patJets.addTagInfos = cms.bool(True)
    #
    ## PU JetID
    process.load("PhysicsTools.PatAlgos.slimming.pileupJetId_cfi")
    process.patJets.userData.userFloats.src = [ cms.InputTag("pileupJetId:fullDiscriminant"), ]

    #VID Electron IDs
    electron_ids = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_CSA14_50ns_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_CSA14_PU20bx25_V0_cff',
                    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV50_CSA14_25ns_cff',
                    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV50_CSA14_startup_cff']
    switchOnVIDElectronIdProducer(process)
    process.egmGsfElectronIDs.physicsObjectSrc = \
        cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.electronIDValueMapProducer.src = \
        cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.electronIDValueMapProducer.ebReducedRecHitCollection = \
        cms.InputTag("reducedEgamma","reducedEBRecHits")
    process.electronIDValueMapProducer.eeReducedRecHitCollection = \
        cms.InputTag("reducedEgamma","reducedEERecHits")
    process.electronIDValueMapProducer.esReducedRecHitCollection = \
        cms.InputTag("reducedEgamma","reducedESRecHits")
    for idmod in electron_ids:
        setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)



def miniAOD_customizeMC(process):
    process.muonMatch.matched = "prunedGenParticles"
    process.electronMatch.matched = "prunedGenParticles"
    process.electronMatch.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.photonMatch.matched = "prunedGenParticles"
    process.photonMatch.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.tauMatch.matched = "prunedGenParticles"
    process.tauGenJets.GenParticles = "prunedGenParticles"
    process.patJetPartonMatch.matched = "prunedGenParticles"
    process.patJetPartonMatch.mcStatus = [ 3, 23 ]
    process.patJetGenJetMatch.matched = "slimmedGenJets"
    process.patMuons.embedGenMatch = False
    process.patElectrons.embedGenMatch = False
    process.patPhotons.embedGenMatch = False
    process.patTaus.embedGenMatch = False
    process.patJets.embedGenPartonMatch = False
    #also jet flavour must be switched to ak4
    process.patJetFlavourAssociation.rParam = 0.4

def miniAOD_customizeOutput(out):
    out.dropMetaData = cms.untracked.string('ALL')
    out.fastCloning= cms.untracked.bool(False)
    out.overrideInputFileSplitLevels = cms.untracked.bool(True)
    out.compressionAlgorithm = cms.untracked.string('LZMA')

def miniAOD_customizeData(process):
    from PhysicsTools.PatAlgos.tools.coreTools import runOnData
    runOnData( process, outputModules = [] )

def miniAOD_customizeAllData(process):
    miniAOD_customizeCommon(process)
    miniAOD_customizeData(process)
    return process

def miniAOD_customizeAllMC(process):
    miniAOD_customizeCommon(process)
    miniAOD_customizeMC(process)
    return process

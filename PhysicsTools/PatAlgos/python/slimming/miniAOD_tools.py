import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

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
    process.patMuons.addPuppiIsolation = cms.bool(True)
    process.patMuons.puppiIsolationChargedHadrons = cms.InputTag("muonPUPPIIsolation","h+-DR040-ThresholdVeto000-ConeVeto000")
    process.patMuons.puppiIsolationNeutralHadrons = cms.InputTag("muonPUPPIIsolation","h0-DR040-ThresholdVeto000-ConeVeto001")
    process.patMuons.puppiIsolationPhotons        = cms.InputTag("muonPUPPIIsolation","gamma-DR040-ThresholdVeto000-ConeVeto001")
    process.patMuons.puppiNoLeptonsIsolationChargedHadrons = cms.InputTag("muonPUPPINoLeptonsIsolation","h+-DR040-ThresholdVeto000-ConeVeto000")
    process.patMuons.puppiNoLeptonsIsolationNeutralHadrons = cms.InputTag("muonPUPPINoLeptonsIsolation","h0-DR040-ThresholdVeto000-ConeVeto001")
    process.patMuons.puppiNoLeptonsIsolationPhotons        = cms.InputTag("muonPUPPINoLeptonsIsolation","gamma-DR040-ThresholdVeto000-ConeVeto001")

    process.patMuons.computeMiniIso = cms.bool(True)
    process.patMuons.computeMuonMVA = cms.bool(True)
    
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
    process.patElectrons.usePfCandidateMultiMap = True
    process.patElectrons.pfCandidateMultiMap    = cms.InputTag("reducedEgamma","reducedGsfElectronPfCandMap")
    process.patElectrons.electronIDSources = cms.PSet(
            # configure many IDs as InputTag <someName> = <someTag> you
            # can comment out those you don't want to save some disk space
            eidRobustLoose      = cms.InputTag("reducedEgamma","eidRobustLoose"),
            eidRobustTight      = cms.InputTag("reducedEgamma","eidRobustTight"),
            eidLoose            = cms.InputTag("reducedEgamma","eidLoose"),
            eidTight            = cms.InputTag("reducedEgamma","eidTight"),
            eidRobustHighEnergy = cms.InputTag("reducedEgamma","eidRobustHighEnergy"),
        )
    process.patElectrons.addPFClusterIso = cms.bool(True)
    #add puppi isolation in miniAOD
    process.patElectrons.addPuppiIsolation = cms.bool(True)
    process.patElectrons.puppiIsolationChargedHadrons = cms.InputTag("egmElectronPUPPIIsolation","h+-DR030-BarVeto000-EndVeto001")
    process.patElectrons.puppiIsolationNeutralHadrons = cms.InputTag("egmElectronPUPPIIsolation","h0-DR030-BarVeto000-EndVeto000")
    process.patElectrons.puppiIsolationPhotons        = cms.InputTag("egmElectronPUPPIIsolation","gamma-DR030-BarVeto000-EndVeto008")
    process.patElectrons.puppiNoLeptonsIsolationChargedHadrons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","h+-DR030-BarVeto000-EndVeto001")
    process.patElectrons.puppiNoLeptonsIsolationNeutralHadrons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","h0-DR030-BarVeto000-EndVeto000")
    process.patElectrons.puppiNoLeptonsIsolationPhotons        = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","gamma-DR030-BarVeto000-EndVeto008")

    process.patElectrons.computeMiniIso = cms.bool(True)

    process.patElectrons.ecalPFClusterIsoMap = cms.InputTag("reducedEgamma", "eleEcalPFClusIso")
    process.patElectrons.hcalPFClusterIsoMap = cms.InputTag("reducedEgamma", "eleHcalPFClusIso")

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
    process.patPhotons.addPFClusterIso = cms.bool(True)

    #add puppi isolation in miniAOD
    process.patPhotons.addPuppiIsolation = cms.bool(True)
    process.patPhotons.puppiIsolationChargedHadrons = cms.InputTag("egmPhotonPUPPIIsolation","h+-DR030-")
    process.patPhotons.puppiIsolationNeutralHadrons = cms.InputTag("egmPhotonPUPPIIsolation","h0-DR030-")
    process.patPhotons.puppiIsolationPhotons        = cms.InputTag("egmPhotonPUPPIIsolation","gamma-DR030-")

    process.patPhotons.ecalPFClusterIsoMap = cms.InputTag("reducedEgamma", "phoEcalPFClusIso")
    process.patPhotons.hcalPFClusterIsoMap = cms.InputTag("reducedEgamma", "phoHcalPFClusIso")
    process.patPhotons.photonSource = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.patPhotons.electronSource = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
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
    process.patOOTPhotons.photonSource = cms.InputTag("reducedEgamma","reducedOOTPhotons")
    process.patOOTPhotons.electronSource = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    #
    process.selectedPatJets.cut = cms.string("pt > 10")
    process.selectedPatMuons.cut = cms.string("pt > 5 || isPFMuon || (pt > 3 && (isGlobalMuon || isStandAloneMuon || numberOfMatches > 0 || muonID('RPCMuLoose')))")
    
    from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
    phase2_muon.toModify(process.selectedPatMuons, cut = "pt > 5 || isPFMuon || (pt > 3 && (isGlobalMuon || isStandAloneMuon || numberOfMatches > 0 || muonID('RPCMuLoose') || muonID('ME0MuonArbitrated') || muonID('GEMMuonArbitrated')) )")
    
    process.selectedPatElectrons.cut = cms.string("")
    process.selectedPatTaus.cut = cms.string("pt > 18. && tauID('decayModeFindingNewDMs')> 0.5")
    process.selectedPatPhotons.cut = cms.string("")

    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

    from PhysicsTools.PatAlgos.slimming.applySubstructure_cff import applySubstructure
    applySubstructure( process )

        
    #
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTriggerStandAlone
    switchOnTriggerStandAlone( process, outputModule = '' )
    process.patTrigger.packTriggerPathNames = cms.bool(True)
    #
    # apply type I + other PFMEt corrections to pat::MET object
    # and estimate systematic uncertainties on MET

    from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncForMiniAODProduction
    runMetCorAndUncForMiniAODProduction(process, metType="PF",
                                        jetCollUnskimmed="patJets")
    
    #caloMET computation
    from PhysicsTools.PatAlgos.tools.metTools import addMETCollection
    addMETCollection(process,
                     labelName = "patCaloMet",
                     metSource = "caloMetM"
                     )

    #noHF pfMET =========

    task = getPatAlgosToolsTask(process)

    process.noHFCands = cms.EDFilter("GenericPFCandidateSelector",
                                     src=cms.InputTag("particleFlow"),
                                     cut=cms.string("abs(pdgId)!=1 && abs(pdgId)!=2 && abs(eta)<3.0")
                                     )
    task.add(process.noHFCands)

    runMetCorAndUncForMiniAODProduction(process,
                                        pfCandColl=cms.InputTag("noHFCands"),
                                        recoMetFromPFCs=True, #needed for HF removal
                                        jetSelection="pt>15 && abs(eta)<3.",
                                        postfix="NoHF"
                                        )

    process.load('PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi')
    task.add(process.slimmedMETs)
    addToProcessAndTask('slimmedMETsNoHF', process.slimmedMETs.clone(), process, task)
    process.slimmedMETsNoHF.src = cms.InputTag("patMETsNoHF")
    process.slimmedMETsNoHF.rawVariation =  cms.InputTag("patPFMetNoHF")
    process.slimmedMETsNoHF.t1Uncertainties = cms.InputTag("patPFMetT1%sNoHF") 
    process.slimmedMETsNoHF.t01Variation = cms.InputTag("patPFMetT0pcT1NoHF")
    process.slimmedMETsNoHF.t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%sNoHF")
    process.slimmedMETsNoHF.tXYUncForRaw = cms.InputTag("patPFMetTxyNoHF")
    process.slimmedMETsNoHF.tXYUncForT1 = cms.InputTag("patPFMetT1TxyNoHF")
    process.slimmedMETsNoHF.tXYUncForT01 = cms.InputTag("patPFMetT0pcT1TxyNoHF")
    process.slimmedMETsNoHF.tXYUncForT1Smear = cms.InputTag("patPFMetT1SmearTxyNoHF")
    process.slimmedMETsNoHF.tXYUncForT01Smear = cms.InputTag("patPFMetT0pcT1SmearTxyNoHF")
    del process.slimmedMETsNoHF.caloMET
    # ================== NoHF pfMET

    #  ==================  CHSMET 
    process.CHSCands = cms.EDFilter("CandPtrSelector",
                                    src=cms.InputTag("packedPFCandidates"),
                                    cut=cms.string("fromPV(0) > 0")
                                    )
    task.add(process.CHSCands)

    process.pfMetCHS = cms.EDProducer("PFMETProducer",
                                      src = cms.InputTag("CHSCands"),
                                      alias = cms.string('pfMet'),
                                      globalThreshold = cms.double(0.0),
                                      calculateSignificance = cms.bool(False),
                                      )
    task.add(process.pfMetCHS)    

    addMETCollection(process,
                     labelName = "patCHSMet",
                     metSource = "pfMetCHS"
                     )

    process.patCHSMet.computeMETSignificance = cms.bool(False)

    #  ==================  CHSMET 

    #  ==================  TrkMET 
    process.TrkCands = cms.EDFilter("CandPtrSelector",
                                    src=cms.InputTag("packedPFCandidates"),
                                    cut=cms.string("charge()!=0 && pvAssociationQuality()>=4 && vertexRef().key()==0")
                                    )
    task.add(process.TrkCands)

    process.pfMetTrk = cms.EDProducer("PFMETProducer",
                                      src = cms.InputTag("TrkCands"),
                                      alias = cms.string('pfMet'),
                                      globalThreshold = cms.double(0.0),
                                      calculateSignificance = cms.bool(False),
                                      )

    task.add(process.pfMetTrk)

    addMETCollection(process,
                     labelName = "patTrkMet",
                     metSource = "pfMetTrk"
                     )

    process.patTrkMet.computeMETSignificance = cms.bool(False)

    #  ==================  TrkMET 
    

    ## PU JetID
    process.load("RecoJets.JetProducers.PileupJetID_cfi")
    task.add(process.pileUpJetIDTask)

    process.patJets.userData.userFloats.src = [ cms.InputTag("pileupJetId:fullDiscriminant"), ]
    process.patJets.userData.userInts.src = [ cms.InputTag("pileupJetId:fullId"), ]

    ## Quark Gluon Likelihood
    process.load('RecoJets.JetProducers.QGTagger_cfi')
    task.add(process.QGTaggerTask)

    process.patJets.userData.userFloats.src += [ cms.InputTag('QGTagger:qgLikelihood'), ]

    ## DeepCSV meta discriminators (simple arithmethic on output probabilities)
    process.load('RecoBTag.Combined.deepFlavour_cff')
    task.add(process.pfDeepCSVDiscriminatorsJetTags)
    process.patJets.discriminatorSources.extend([
            cms.InputTag('pfDeepCSVDiscriminatorsJetTags:BvsAll' ),
            cms.InputTag('pfDeepCSVDiscriminatorsJetTags:CvsB'   ),
            cms.InputTag('pfDeepCSVDiscriminatorsJetTags:CvsL'   ),
            ])

    ## CaloJets
    process.caloJetMap = cms.EDProducer("RecoJetDeltaRValueMapProducer",
         src = process.patJets.jetSource,
         matched = cms.InputTag("ak4CaloJets"),
         distMax = cms.double(0.4),
         values = cms.vstring('pt','emEnergyFraction'),
	 valueLabels = cms.vstring('pt','emEnergyFraction'),
	 lazyParser = cms.bool(True) )
    task.add(process.caloJetMap)
    process.patJets.userData.userFloats.src += [ cms.InputTag("caloJetMap:pt"), cms.InputTag("caloJetMap:emEnergyFraction") ]

    #Muon object modifications 
    from PhysicsTools.PatAlgos.slimming.muonIsolationsPUPPI_cfi import makeInputForPUPPIIsolationMuon
    makeInputForPUPPIIsolationMuon(process)

    #EGM object modifications 
    from PhysicsTools.PatAlgos.slimming.egmIsolationsPUPPI_cfi import makeInputForPUPPIIsolationEgm
    makeInputForPUPPIIsolationEgm(process)
    from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import egamma_modifications
    process.slimmedElectrons.modifierConfig.modifications = egamma_modifications
    process.slimmedPhotons.modifierConfig.modifications   = egamma_modifications

    #VID Electron IDs
    electron_ids = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_PHYS14_PU20bx25_V2_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_50ns_V2_cff',
                    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV60_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_Trig_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_50ns_Trig_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff', 
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff', 
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff',
                    ]
    switchOnVIDElectronIdProducer(process,DataFormat.MiniAOD, task)
    process.egmGsfElectronIDs.physicsObjectSrc = \
        cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.electronMVAValueMapProducer.src = \
        cms.InputTag('reducedEgamma','reducedGedGsfElectrons')
    process.electronRegressionValueMapProducer.src = \
        cms.InputTag('reducedEgamma','reducedGedGsfElectrons')
    for idmod in electron_ids:
        setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection,None,False,task)
        
    #heepIDVarValueMaps only exists if HEEP V6.1 or HEEP 7.0 ID has already been loaded
    if hasattr(process,'heepIDVarValueMaps'):
        process.heepIDVarValueMaps.elesMiniAOD = cms.InputTag('reducedEgamma','reducedGedGsfElectrons')
        #force HEEP to use miniAOD (otherwise it'll detect the AOD)
        process.heepIDVarValueMaps.dataFormat = cms.int32(2)
        
        #add the HEEP trk isol to the slimmed electron, add it to the first FromFloatValMap modifier
        for pset in process.slimmedElectrons.modifierConfig.modifications:
            if pset.hasParameter("modifierName") and pset.modifierName == cms.string('EGExtraInfoModifierFromFloatValueMaps'):
                pset.electron_config.heepTrkPtIso = cms.InputTag("heepIDVarValueMaps","eleTrkPtIso")
                break

    #VID Photon IDs
    photon_ids = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_25ns_V1_cff',
                  'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_50ns_V1_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V2p1_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V2p1_cff',
                  'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V1_TrueVtx_cff',
                  'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V2_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff', 
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff',
                  'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring16_V2p2_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff']
    switchOnVIDPhotonIdProducer(process,DataFormat.AOD, task) 
    process.egmPhotonIsolation.srcToIsolate = \
        cms.InputTag("reducedEgamma","reducedGedPhotons")  
    for iPSet in process.egmPhotonIsolation.isolationConeDefinitions:
        iPSet.particleBasedIsolation = cms.InputTag("reducedEgamma","reducedPhotonPfCandMap")    

    process.egmPhotonIDs.physicsObjectSrc = \
        cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.photonIDValueMapProducer.src = \
        cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.photonRegressionValueMapProducer.src = \
        cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.photonIDValueMapProducer.particleBasedIsolation = \
        cms.InputTag("reducedEgamma","reducedPhotonPfCandMap")
    process.photonMVAValueMapProducer.src = \
        cms.InputTag('reducedEgamma','reducedGedPhotons')
    for idmod in photon_ids:
        setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection,None,False,task)

    #add the cut base IDs bitmaps of which cuts passed
    from RecoEgamma.EgammaTools.egammaObjectModifications_tools import makeVIDBitsModifier
    egamma_modifications.append(makeVIDBitsModifier(process,"egmGsfElectronIDs","egmPhotonIDs"))

    #-- Adding boosted taus
    from RecoTauTag.Configuration.boostedHPSPFTaus_cfi import addBoostedTaus
    addBoostedTaus(process)
    process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
    process.load("RecoTauTag.Configuration.HPSPFTaus_cff")
    #-- Adding customization for 94X 2017 legacy reMniAOD
    from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
    _makePatTausTaskWithRetrainedMVATauID = process.makePatTausTask.copy()
    _makePatTausTaskWithRetrainedMVATauID.add(process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTTask)
    run2_miniAOD_94XFall17.toReplaceWith(
        process.makePatTausTask, _makePatTausTaskWithRetrainedMVATauID
        )
    #-- Adding custimization for 80X 2016 legacy reMiniAOD
    from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
    _makePatTausTaskWithTauReReco = process.makePatTausTask.copy()
    _makePatTausTaskWithTauReReco.add(process.PFTauTask)
    run2_miniAOD_80XLegacy.toReplaceWith(
        process.makePatTausTask, _makePatTausTaskWithTauReReco)

    # Adding puppi jets
    if not hasattr(process, 'ak4PFJetsPuppi'): #MM: avoid confilct with substructure call
        process.load('RecoJets.JetProducers.ak4PFJetsPuppi_cfi')
        task.add(process.ak4PFJets)
        task.add(process.ak4PFJetsPuppi)
    process.ak4PFJetsPuppi.doAreaFastjet = True # even for standard ak4PFJets this is overwritten in RecoJets/Configuration/python/RecoPFJets_cff
    from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import j2tParametersVX
    process.ak4PFJetsPuppiTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
        j2tParametersVX,
        jets = cms.InputTag("ak4PFJetsPuppi")
    )
    task.add(process.ak4PFJetsPuppiTracksAssociatorAtVertex)
    process.patJetPuppiCharge = cms.EDProducer("JetChargeProducer",
        src = cms.InputTag("ak4PFJetsPuppiTracksAssociatorAtVertex"),
        var = cms.string('Pt'),
        exp = cms.double(1.0)
    )
    task.add(process.patJetPuppiCharge)

    noDeepFlavourDiscriminators = [x.value() for x in process.patJets.discriminatorSources if not "DeepFlavour" in x.value()]
    addJetCollection(process, postfix   = "", labelName = 'Puppi', jetSource = cms.InputTag('ak4PFJetsPuppi'),
                    jetCorrections = ('AK4PFPuppi', ['L2Relative', 'L3Absolute'], ''),
                    pfCandidates = cms.InputTag('puppi'), # using Puppi candidates as input for b tagging of Puppi jets
                    algo= 'AK', rParam = 0.4, btagDiscriminators = noDeepFlavourDiscriminators
                    )
    
    process.patJetGenJetMatchPuppi.matched = 'slimmedGenJets'
    
    process.patJetsPuppi.jetChargeSource = cms.InputTag("patJetPuppiCharge")

    process.selectedPatJetsPuppi.cut = cms.string("pt > 15")

    from PhysicsTools.PatAlgos.slimming.applyDeepBtagging_cff import applyDeepBtagging
    applyDeepBtagging( process )

    addToProcessAndTask('slimmedJetsPuppiNoMultiplicities', process.slimmedJetsNoDeepFlavour.clone(), process, task)
    process.slimmedJetsPuppiNoMultiplicities.src = cms.InputTag("selectedPatJetsPuppi")
    process.slimmedJetsPuppiNoMultiplicities.packedPFCandidates = cms.InputTag("packedPFCandidates")

    from PhysicsTools.PatAlgos.patPuppiJetSpecificProducer_cfi import patPuppiJetSpecificProducer
    from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
    process.patPuppiJetSpecificProducer = patPuppiJetSpecificProducer.clone(
      src=cms.InputTag("slimmedJetsPuppiNoMultiplicities"),
    )
    task.add(process.patPuppiJetSpecificProducer)
    updateJetCollection(
       process,
       labelName = 'PuppiJetSpecific',
       jetSource = cms.InputTag('slimmedJetsPuppiNoMultiplicities'),
    )
    process.updatedPatJetsPuppiJetSpecific.userData.userFloats.src = ['patPuppiJetSpecificProducer:puppiMultiplicity', 'patPuppiJetSpecificProducer:neutralPuppiMultiplicity', 'patPuppiJetSpecificProducer:neutralHadronPuppiMultiplicity', 'patPuppiJetSpecificProducer:photonPuppiMultiplicity', 'patPuppiJetSpecificProducer:HFHadronPuppiMultiplicity', 'patPuppiJetSpecificProducer:HFEMPuppiMultiplicity' ]
    process.slimmedJetsPuppi = process.selectedUpdatedPatJetsPuppiJetSpecific.clone()
    delattr(process, 'selectedUpdatedPatJetsPuppiJetSpecific')

    task.add(process.slimmedJetsPuppi)
    
    ## puppi met
    from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppies
    makePuppies( process );

    runMetCorAndUncForMiniAODProduction(process, metType="Puppi",
                                        pfCandColl=cms.InputTag("puppiForMET"),
                                        jetCollUnskimmed="slimmedJetsPuppi",
                                        recoMetFromPFCs=True,
                                        jetFlavor="AK4PFPuppi",
                                        postfix="Puppi"
                                        )
    
    process.load('PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi')
    task.add(process.slimmedMETs)
    addToProcessAndTask('slimmedMETsPuppi', process.slimmedMETs.clone(), process, task)
    process.slimmedMETsPuppi.src = cms.InputTag("patMETsPuppi")
    process.slimmedMETsPuppi.rawVariation =  cms.InputTag("patPFMetPuppi")
    process.slimmedMETsPuppi.t1Uncertainties = cms.InputTag("patPFMetT1%sPuppi")
    process.slimmedMETsPuppi.t01Variation = cms.InputTag("patPFMetT0pcT1Puppi")
    process.slimmedMETsPuppi.t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%sPuppi")
    process.slimmedMETsPuppi.tXYUncForRaw = cms.InputTag("patPFMetTxyPuppi")
    process.slimmedMETsPuppi.tXYUncForT1 = cms.InputTag("patPFMetT1TxyPuppi")
    process.slimmedMETsPuppi.tXYUncForT01 = cms.InputTag("patPFMetT0pcT1TxyPuppi")
    process.slimmedMETsPuppi.tXYUncForT1Smear = cms.InputTag("patPFMetT1SmearTxyPuppi")
    process.slimmedMETsPuppi.tXYUncForT01Smear = cms.InputTag("patPFMetT0pcT1SmearTxyPuppi")
    del process.slimmedMETsPuppi.caloMET

    # add DetIdAssociatorRecords to EventSetup (for isolatedTracks)
    process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")


def miniAOD_customizeMC(process):
    task = getPatAlgosToolsTask(process)
    #GenJetFlavourInfos
    process.load("PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi")
    task.add(process.selectedHadronsAndPartons)
    task.add(process.selectedHadronsAndPartonsForGenJetsFlavourInfos)
    
    process.load("PhysicsTools.JetMCAlgos.AK4GenJetFlavourInfos_cfi")
    task.add(process.ak4GenJetFlavourInfos)

    process.load('PhysicsTools.PatAlgos.slimming.slimmedGenJetsFlavourInfos_cfi')
    task.add(process.slimmedGenJetsFlavourInfos)

    #slimmed pileup information
    process.load('PhysicsTools.PatAlgos.slimming.slimmedAddPileupInfo_cfi')
    task.add(process.slimmedAddPileupInfo)

    process.muonMatch.matched = "prunedGenParticles"
    process.electronMatch.matched = "prunedGenParticles"
    process.electronMatch.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.photonMatch.matched = "prunedGenParticles"
    process.photonMatch.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.ootPhotonMatch.matched = "prunedGenParticles"
    process.ootPhotonMatch.src = cms.InputTag("reducedEgamma","reducedOOTPhotons")
    process.tauMatch.matched = "prunedGenParticles"
    process.tauGenJets.GenParticles = "prunedGenParticles"
    #Boosted taus 
    process.tauMatchBoosted.matched = "prunedGenParticles"
    process.tauGenJetsBoosted.GenParticles = "prunedGenParticles"
    process.patJetPartons.particles = "genParticles"
    process.patJetPartonMatch.matched = "prunedGenParticles"
    process.patJetPartonMatch.mcStatus = [ 3, 23 ]
    process.patJetGenJetMatch.matched = "slimmedGenJets"
    process.patJetGenJetMatchAK8.matched =  "slimmedGenJetsAK8"
    process.patMuons.embedGenMatch = False
    process.patElectrons.embedGenMatch = False
    process.patPhotons.embedGenMatch = False
    process.patOOTPhotons.embedGenMatch = False
    process.patTaus.embedGenMatch = False
    process.patTausBoosted.embedGenMatch = False
    process.patJets.embedGenPartonMatch = False
    #also jet flavour must be switched
    process.patJetFlavourAssociation.rParam = 0.4

def miniAOD_customizeOutput(out):
    from PhysicsTools.PatAlgos.slimming.MicroEventContent_cff import MiniAODOverrideBranchesSplitLevel
    out.overrideBranchesSplitLevel = MiniAODOverrideBranchesSplitLevel
    out.splitLevel = cms.untracked.int32(0)
    out.dropMetaData = cms.untracked.string('ALL')
    out.fastCloning= cms.untracked.bool(False)
    out.overrideInputFileSplitLevels = cms.untracked.bool(True)
    out.compressionAlgorithm = cms.untracked.string('LZMA')

def miniAOD_customizeData(process):
    from PhysicsTools.PatAlgos.tools.coreTools import runOnData
    runOnData( process, outputModules = [] )
    process.load("RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff")
    task = getPatAlgosToolsTask(process)
    task.add(process.ctppsLocalTrackLiteProducer)

def miniAOD_customizeAllData(process):
    miniAOD_customizeCommon(process)
    miniAOD_customizeData(process)
    return process

def miniAOD_customizeAllMC(process):
    miniAOD_customizeCommon(process)
    miniAOD_customizeMC(process)
    return process

def miniAOD_customizeAllMCFastSim(process):
    miniAOD_customizeCommon(process)
    miniAOD_customizeMC(process)
    from PhysicsTools.PatAlgos.slimming.metFilterPaths_cff import miniAOD_customizeMETFiltersFastSim
    process = miniAOD_customizeMETFiltersFastSim(process)
    from PhysicsTools.PatAlgos.slimming.isolatedTracks_cfi import miniAOD_customizeIsolatedTracksFastSim
    process = miniAOD_customizeIsolatedTracksFastSim(process)
    return process

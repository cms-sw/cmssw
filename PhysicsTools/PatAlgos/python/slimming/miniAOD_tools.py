import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

from PhysicsTools.PatUtils.tools.pfforTrkMET_cff import *

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

    process.patMuons.computeMiniIso = True
    process.patMuons.computeMuonMVA = True
    process.patMuons.computeMuonIDMVA = True
    process.patMuons.computeSoftMuonMVA = True

    process.patMuons.addTriggerMatching = True
    from Configuration.Eras.Modifier_run2_muon_2016_cff import run2_muon_2016
    from Configuration.Eras.Modifier_run2_muon_2017_cff import run2_muon_2017
    from Configuration.Eras.Modifier_run2_muon_2018_cff import run2_muon_2018
    run2_muon_2016.toModify( process.patMuons, effectiveAreaVec = [0.0735,0.0619,0.0465,0.0433,0.0577])
    run2_muon_2017.toModify( process.patMuons, effectiveAreaVec = [0.0566, 0.0562, 0.0363, 0.0119, 0.0064])
    run2_muon_2018.toModify( process.patMuons, effectiveAreaVec = [0.0566, 0.0562, 0.0363, 0.0119, 0.0064])
    run2_muon_2016.toModify( process.patMuons, mvaTrainingFile = "RecoMuon/MuonIdentification/data/mu_2016_BDTG.weights.xml")

    process.patMuons.computePuppiCombinedIso = True
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
    process.patElectrons.electronIDSources = cms.PSet()

    from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
    from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
    (run2_miniAOD_80XLegacy | run2_miniAOD_94XFall17).toModify(process.patElectrons,
                                                               addPFClusterIso = True,
                                                               ecalPFClusterIsoMap = "reducedEgamma:eleEcalPFClusIso",
                                                               hcalPFClusterIsoMap = "reducedEgamma:eleHcalPFClusIso")

    #add puppi isolation in miniAOD
    process.patElectrons.addPuppiIsolation = cms.bool(True)
    process.patElectrons.puppiIsolationChargedHadrons = cms.InputTag("egmElectronPUPPIIsolation","h+-DR030-BarVeto000-EndVeto001")
    process.patElectrons.puppiIsolationNeutralHadrons = cms.InputTag("egmElectronPUPPIIsolation","h0-DR030-BarVeto000-EndVeto000")
    process.patElectrons.puppiIsolationPhotons        = cms.InputTag("egmElectronPUPPIIsolation","gamma-DR030-BarVeto000-EndVeto008")
    process.patElectrons.puppiNoLeptonsIsolationChargedHadrons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","h+-DR030-BarVeto000-EndVeto001")
    process.patElectrons.puppiNoLeptonsIsolationNeutralHadrons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","h0-DR030-BarVeto000-EndVeto000")
    process.patElectrons.puppiNoLeptonsIsolationPhotons        = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","gamma-DR030-BarVeto000-EndVeto008")

    process.patElectrons.computeMiniIso = cms.bool(True)

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

    #add puppi isolation in miniAOD
    process.patPhotons.addPuppiIsolation = cms.bool(True)
    process.patPhotons.puppiIsolationChargedHadrons = cms.InputTag("egmPhotonPUPPIIsolation","h+-DR030-")
    process.patPhotons.puppiIsolationNeutralHadrons = cms.InputTag("egmPhotonPUPPIIsolation","h0-DR030-")
    process.patPhotons.puppiIsolationPhotons        = cms.InputTag("egmPhotonPUPPIIsolation","gamma-DR030-")

    (run2_miniAOD_80XLegacy | run2_miniAOD_94XFall17).toModify(process.patPhotons,
                                                               addPFClusterIso = True,
                                                               ecalPFClusterIsoMap = "reducedEgamma:phoEcalPFClusIso",
                                                               hcalPFClusterIsoMap = "reducedEgamma:phoHcalPFClusIso")
    #the 80X legacy customsations are done in ootPhotonProducer for OOT photons
    run2_miniAOD_94XFall17.toModify(process.patOOTPhotons,
                                    addPFClusterIso = True,
                                    ecalPFClusterIsoMap = "reducedEgamma:ootPhoEcalPFClusIso",
                                    hcalPFClusterIsoMap = "reducedEgamma:ootPhoHcalPFClusIso")


    process.patPhotons.photonSource = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.patPhotons.electronSource = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    
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
    from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
    pp_on_AA.toModify(process.selectedPatMuons, cut = "pt > 5 || isPFMuon || (pt > 1.2 && (isGlobalMuon || isStandAloneMuon) )")

    process.selectedPatElectrons.cut = cms.string("")
    process.selectedPatTaus.cut = cms.string("pt > 18. && tauID('decayModeFindingNewDMs')> 0.5")
    process.selectedPatPhotons.cut = cms.string("")

    _dummyPatJets = process.selectedPatJets.clone(cut = "pt < 0")
    task = getPatAlgosToolsTask(process)

    def _applySubstructure(process):
        from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
        
        from PhysicsTools.PatAlgos.slimming.applySubstructure_cff import applySubstructure
        applySubstructure( process )
    (~pp_on_AA).toModify(process, _applySubstructure)

    pp_on_AA.toModify(process, func = lambda p: addToProcessAndTask('slimmedJets', p.selectedPatJets.clone(), p, task))
    pp_on_AA.toModify(process, func = lambda p: addToProcessAndTask('slimmedJetsAK8', _dummyPatJets.clone(), p, task))

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
    (~pp_on_AA).toModify(process.slimmedMETs, addDeepMETs = True)

    def _add_slimmedMETsNoHF(process):
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
    (~pp_on_AA).toModify(process, _add_slimmedMETsNoHF)
    # ================== NoHF pfMET

    #  ==================  CHSMET 
    process.load("CommonTools.ParticleFlow.pfCHS_cff")
    task.add(process.pfCHS)

    from RecoMET.METProducers.pfMet_cfi import pfMet
    process.pfMetCHS = pfMet.clone(src = 'pfCHS')
    task.add(process.pfMetCHS)

    addMETCollection(process,
                     labelName = "patCHSMet",
                     metSource = "pfMetCHS"
                     )

    process.patCHSMet.computeMETSignificance = cms.bool(False)

    #  ==================  CHSMET 

    #  ==================  TrkMET 
    process.TrkCands = chargedPackedCandsForTkMet.clone()
    task.add(process.TrkCands)

    process.pfMetTrk = pfMet.clone(src = 'TrkCands')
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

    process.patJets.userData.userFloats.src += [ 'QGTagger:qgLikelihood', ]

    #HF jet shower shape
    process.load('RecoJets.JetProducers.hfJetShowerShape_cfi')
    task.add(process.hfJetShowerShape)

    process.patJets.userData.userFloats.src += [ 'hfJetShowerShape:sigmaEtaEta', 'hfJetShowerShape:sigmaPhiPhi']
    process.patJets.userData.userInts.src += [ 'hfJetShowerShape:centralEtaStripSize', 'hfJetShowerShape:adjacentEtaStripsSize']

    ## DeepCSV meta discriminators (simple arithmethic on output probabilities)
    def _add_deepFlavour(process):
        process.load('RecoBTag.Combined.deepFlavour_cff')
        task.add(process.pfDeepCSVDiscriminatorsJetTags)
        process.patJets.discriminatorSources.extend([
            'pfDeepCSVDiscriminatorsJetTags:BvsAll',
            'pfDeepCSVDiscriminatorsJetTags:CvsB',
            'pfDeepCSVDiscriminatorsJetTags:CvsL',
        ])
    (~pp_on_AA).toModify(process, _add_deepFlavour)

    ## CaloJets
    process.caloJetMap = cms.EDProducer("RecoJetDeltaRValueMapProducer",
         src = process.patJets.jetSource,
         matched = cms.InputTag("ak4CaloJets"),
         distMax = cms.double(0.4),
         values = cms.vstring('pt','emEnergyFraction'),
	 valueLabels = cms.vstring('pt','emEnergyFraction'),
	 lazyParser = cms.bool(True) )
    task.add(process.caloJetMap)
    process.patJets.userData.userFloats.src += [ 'caloJetMap:pt', 'caloJetMap:emEnergyFraction' ]

    pp_on_AA.toModify(process.patJets.userData.userInts, src = [] )
    pp_on_AA.toModify(process.patJets.userData.userFloats, src = [] )

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
    process.patElectrons.addElectronID = cms.bool(True)
    electron_ids = ['RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
                    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV71_cff',
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
    process.egmGsfElectronIDs.physicsObjectSrc = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.electronMVAValueMapProducer.src = cms.InputTag('reducedEgamma','reducedGedGsfElectrons')

    # To use older DataFormats, the electronMVAValueMapProducer MUST take a updated electron collection
    # such that the conversion variables are filled correctly.
    process.load("RecoEgamma.EgammaTools.gedGsfElectronsTo106X_cff")
    run2_miniAOD_80XLegacy.toModify(task, func=lambda t: t.add(process.gedGsfElectronsFrom80XTo106XTask))
    run2_miniAOD_80XLegacy.toModify(process.electronMVAValueMapProducer,
                                     keysForValueMaps = cms.InputTag('reducedEgamma','reducedGedGsfElectrons'),
                                     src = cms.InputTag("gedGsfElectronsFrom80XTo106X"))

    run2_miniAOD_94XFall17.toModify(task, func=lambda t: t.add(process.gedGsfElectronsFrom94XTo106XTask))
    run2_miniAOD_94XFall17.toModify(process.electronMVAValueMapProducer,
                                     keysForValueMaps = cms.InputTag('reducedEgamma','reducedGedGsfElectrons'),
                                     src = cms.InputTag("gedGsfElectronsFrom94XTo106X"))

    from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
    pp_on_AA.toModify(task, func=lambda t: t.add(process.gedGsfElectronsFrom94XTo106XTask))
    pp_on_AA.toModify(process.electronMVAValueMapProducer,
                                     keysForValueMaps = cms.InputTag('reducedEgamma','reducedGedGsfElectrons'),
                                     src = "gedGsfElectronsFrom94XTo106X")

    for idmod in electron_ids:
        setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection,None,False,task)

    #VID Photon IDs
    process.patPhotons.addPhotonID = cms.bool(True)
    photon_ids = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V1_TrueVtx_cff',
                  'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V2_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff', 
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff',
                  'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring16_V2p2_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff']
    switchOnVIDPhotonIdProducer(process,DataFormat.AOD, task) 
    process.egmPhotonIDs.physicsObjectSrc = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.photonMVAValueMapProducer.src = cms.InputTag('reducedEgamma','reducedGedPhotons')
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
    _makePatTausTaskWithRetrainedMVATauID = process.makePatTausTask.copy()
    _makePatTausTaskWithRetrainedMVATauID.add(process.hpsPFTauBasicDiscriminatorsTask,
                                              process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTTask,
                                              process.hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTTask,
                                              process.hpsPFTauBasicDiscriminatorsdR03Task,
                                              process.hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTTask,
                                              process.hpsPFTauDiscriminationByMVA6rawElectronRejection,
                                              process.hpsPFTauDiscriminationByMVA6ElectronRejection,
                                              process.hpsPFTauDiscriminationByMuonRejection3)
    from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
    (run2_miniAOD_94XFall17 | run2_miniAOD_UL).toReplaceWith(
        process.makePatTausTask, _makePatTausTaskWithRetrainedMVATauID
        )
    #-- Adding DeepTauID
    # deepTau v2p1 and v2p5
    _updatedTauName = 'slimmedTausDeepIDsv2p1'
    _noUpdatedTauName = 'slimmedTausNoDeepIDs'
    import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
    tauIdEmbedder = tauIdConfig.TauIDEmbedder(
        process, debug = False,
        originalTauName = _noUpdatedTauName,
        updatedTauName = _updatedTauName,
        toKeep = ['deepTau2017v2p1','deepTau2018v2p5']
    )
    from Configuration.Eras.Modifier_phase2_common_cff import phase2_common #Phase2 Tau MVA
    phase2_common.toModify(tauIdEmbedder.toKeep, func=lambda t:t.append('newDMPhase2v1')) #Phase2 Tau isolation MVA
    phase2_common.toModify(tauIdEmbedder.toKeep, func=lambda t:t.append('againstElePhase2v1')) #Phase2 Tau anti-e MVA
    tauIdEmbedder.runTauID()
    addToProcessAndTask(_noUpdatedTauName, process.slimmedTaus.clone(),process,task)
    delattr(process, 'slimmedTaus')
    process.slimmedTaus = getattr(process, _updatedTauName).clone()
    process.rerunMvaIsolationTask.add(process.slimmedTaus)
    task.add(process.rerunMvaIsolationTask)

    #-- Rerun tauID against dead ECal towers to taus for the various re-MiniAOD eras
    # to enable default behoviour with leading track extrapolation to ECAL
    _makePatTausTaskWithDeadECalVeto = process.makePatTausTask.copy()
    _makePatTausTaskWithDeadECalVeto.add(
        process.hpsPFTauDiscriminationByDeadECALElectronRejection
    )
    _run2_miniAOD_ANY = (run2_miniAOD_80XLegacy | run2_miniAOD_94XFall17 | run2_miniAOD_UL)
    _run2_miniAOD_ANY.toReplaceWith(
        process.makePatTausTask, _makePatTausTaskWithDeadECalVeto
    )

    #-- Adding customization for 80X 2016 legacy reMiniAOD and 2018 heavy ions
    _makePatTausTaskWithTauReReco = process.makePatTausTask.copy()
    _makePatTausTaskWithTauReReco.add(process.PFTauTask)
    (run2_miniAOD_80XLegacy | pp_on_AA).toReplaceWith(
        process.makePatTausTask, _makePatTausTaskWithTauReReco
        )
    
    # Adding puppi jets
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    process.load('RecoJets.JetProducers.ak4PFJets_cfi')
    from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
    _rerun_puppijets_task = task.copy()
    _rerun_puppijets_task.add(process.puppi, process.ak4PFJetsPuppi)
    (_run2_miniAOD_ANY | pA_2016 | pp_on_AA).toReplaceWith(task, _rerun_puppijets_task)

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

    def _add_jetsPuppi(process):
        from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
        noDeepFlavourDiscriminators = [x.value() if isinstance(x, cms.InputTag) else x for x in process.patJets.discriminatorSources 
                                       if not "DeepFlavour" in str(x)]
        addJetCollection(process, postfix   = "", labelName = 'Puppi', jetSource = cms.InputTag('ak4PFJetsPuppi'),
                         jetCorrections = ('AK4PFPuppi', ['L2Relative', 'L3Absolute'], ''),
                         pfCandidates = cms.InputTag("particleFlow"),
                         algo= 'AK', rParam = 0.4, btagDiscriminators = noDeepFlavourDiscriminators
                     )

        process.patJetGenJetMatchPuppi.matched = 'slimmedGenJets'
    
        process.patJetsPuppi.jetChargeSource = cms.InputTag("patJetPuppiCharge")
    
        process.selectedPatJetsPuppi.cut = cms.string("pt > 10")
    
        from PhysicsTools.PatAlgos.slimming.applyDeepBtagging_cff import applyDeepBtagging
        applyDeepBtagging( process )

        process.slimmedJetsNoDeepFlavour.dropTagInfos = '0'
        process.updatedPatJetsTransientCorrectedSlimmedDeepFlavour.addTagInfos = True
        process.updatedPatJetsTransientCorrectedSlimmedDeepFlavour.tagInfoSources = ["pixelClusterTagInfos"]
        _run2_miniAOD_ANY.toModify(process.updatedPatJetsTransientCorrectedSlimmedDeepFlavour, addTagInfos = False )
    (~pp_on_AA).toModify(process, _add_jetsPuppi)

    pp_on_AA.toModify(process, func = lambda p: addToProcessAndTask('slimmedJetsPuppi', _dummyPatJets.clone(), p, task))

    # Embed pixelClusterTagInfos in slimmedJets
    process.patJets.addTagInfos = True
    process.patJets.tagInfoSources = ["pixelClusterTagInfos"]

    _run2_miniAOD_ANY.toModify(process.patJets, addTagInfos = False )
    
    pp_on_AA.toModify(process.patJets, tagInfoSources = cms.VInputTag(["impactParameterTagInfos","secondaryVertexTagInfos"]) )

    ## puppi met
    def _add_metPuppi(process):
        process.load('RecoMET.METProducers.pfMetPuppi_cfi')
        _rerun_puppimet_task = task.copy()
        _rerun_puppimet_task.add(process.puppiNoLep, process.pfMetPuppi)
        (_run2_miniAOD_ANY | pA_2016 | pp_on_AA).toReplaceWith(task, _rerun_puppimet_task)
    
        runMetCorAndUncForMiniAODProduction(process, metType="Puppi",
                                            jetCollUnskimmed="slimmedJetsPuppi",
                                            recoMetFromPFCs=True,
                                            jetFlavor="AK4PFPuppi",
                                            postfix="Puppi"
                                        )
    (~pp_on_AA).toModify(process, _add_metPuppi)

    process.load('PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi')
    task.add(process.slimmedMETs)

    def _add_slimmedMETsPuppi(process):
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
    (~pp_on_AA).toModify(process, _add_slimmedMETsPuppi)

    def _add_deepMET(process):
        from RecoMET.METPUSubtraction.deepMETProducer_cff import deepMETsResolutionTune, deepMETsResponseTune

        addToProcessAndTask('deepMETsResolutionTune', deepMETsResolutionTune, process, task)
        addToProcessAndTask('deepMETsResponseTune', deepMETsResponseTune, process, task)
    (~pp_on_AA).toModify(process, _add_deepMET)

    # add DetIdAssociatorRecords to EventSetup (for isolatedTracks)
    process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")

    # EGamma objects from HGCal are not yet in GED
    # so add companion collections for Phase-II MiniAOD production
    from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
    process.load("RecoEgamma.EgammaTools.slimmedEgammaHGC_cff")
    phase2_hgcal.toModify(task, func=lambda t: t.add(process.slimmedEgammaHGCTask))

    # L1 pre-firing weights for 2016, 2017, and 2018
    from Configuration.Eras.Modifier_run2_L1prefiring_cff import run2_L1prefiring
    from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
    from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
    from Configuration.Eras.Modifier_stage2L1Trigger_2018_cff import stage2L1Trigger_2018
    from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
    process.load("PhysicsTools.PatUtils.L1PrefiringWeightProducer_cff")
    (stage2L1Trigger & tracker_apv_vfp30_2016).toModify(process.prefiringweight, DataEraECAL = "UL2016preVFP", DataEraMuon = "2016preVFP" )
    (stage2L1Trigger & ~tracker_apv_vfp30_2016).toModify(process.prefiringweight, DataEraECAL = "UL2016postVFP", DataEraMuon = "2016postVFP" )
    stage2L1Trigger_2017.toModify(process.prefiringweight, DataEraECAL = "UL2017BtoF", DataEraMuon = "20172018")
    stage2L1Trigger_2018.toModify(process.prefiringweight, DataEraECAL = "None", DataEraMuon = "20172018")
    run2_L1prefiring.toModify(task, func=lambda t: t.add(process.prefiringweight))

    from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJetSetup import removeL1FastJetJECs
    pp_on_AA.toModify(process, removeL1FastJetJECs)

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
    pp_on_AA.toModify(process.patJetPartonMatch, matched =  "hiSignalGenParticles")
    from Configuration.ProcessModifiers.genJetSubEvent_cff import genJetSubEvent
    genJetSubEvent.toModify(process.patJetPartonMatch, matched =  "cleanedPartons")
    process.patJetPartonMatch.mcStatus = [ 3, 23 ]
    process.patJetGenJetMatch.matched = "slimmedGenJets"
    (~pp_on_AA).toModify(process, patJetGenJetMatchAK8Puppi = dict(matched =  "slimmedGenJetsAK8"))
    process.patMuons.embedGenMatch = False
    process.patElectrons.embedGenMatch = False
    process.patPhotons.embedGenMatch = False
    process.patOOTPhotons.embedGenMatch = False
    process.patTaus.embedGenMatch = False
    process.patTausBoosted.embedGenMatch = False
    process.patJets.embedGenPartonMatch = False
    #also jet flavour must be switched
    process.patJetFlavourAssociation.rParam = 0.4
    
    from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJetSetup import removeJECsForMC
    pp_on_AA.toModify(process, removeJECsForMC)
    pp_on_AA.toReplaceWith(task,task.copyAndExclude([process.slimmedGenJetsFlavourInfos]))


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
    process.load("RecoPPS.Local.ctppsLocalTrackLiteProducer_cff")
    process.load("RecoPPS.ProtonReconstruction.ctppsProtons_cff")
    process.load("Geometry.VeryForwardGeometry.geometryRPFromDB_cfi")
    process.load('L1Trigger.L1TGlobal.simGtExtFakeProd_cfi')
    task = getPatAlgosToolsTask(process)
    from Configuration.Eras.Modifier_ctpps_cff import ctpps
    ctpps.toModify(task, func=lambda t: t.add(process.ctppsLocalTrackLiteProducer))
    ctpps.toModify(task, func=lambda t: t.add(process.ctppsProtons))
    from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
    run2_miniAOD_UL.toModify(task, func=lambda t: t.add(process.simGtExtUnprefireable))

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
    process.patMuons.addTriggerMatching = False
    # Disable pixelClusterTagInfos in FastSim (no siPixelCluster available)
    from Configuration.Eras.Modifier_fastSim_cff import fastSim
    fastSim.toModify(process.patJets, addTagInfos = cms.bool(False) )
    fastSim.toModify(process.slimmedJetsNoDeepFlavour, dropTagInfos = cms.string('1') )
    fastSim.toModify(process.updatedPatJetsSlimmedDeepFlavour, addTagInfos = cms.bool(False) )
    fastSim.toModify(process.updatedPatJetsTransientCorrectedSlimmedDeepFlavour, addTagInfos = cms.bool(False) )

    return process

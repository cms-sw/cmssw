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
    process.patElectrons.addPFClusterIso = cms.bool(True)
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
    process.selectedPatJets.cut = cms.string("pt > 10")
    process.selectedPatMuons.cut = cms.string("pt > 5 || isPFMuon || (pt > 3 && (isGlobalMuon || isStandAloneMuon || numberOfMatches > 0 || muonID('RPCMuLoose')))")
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
                                        jetCollUnskimmed="patJets",
                                        jetColl="selectedPatJets")
    
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
    runMetCorAndUncForMiniAODProduction(process,
                                        pfCandColl=cms.InputTag("noHFCands"),
                                        recomputeMET=True, #needed for HF removal
                                        postfix="NoHF"
                                        )
    process.load('PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi')
    process.slimmedMETsNoHF = process.slimmedMETs.clone()
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

    #keep this after all addJetCollections otherwise it will attempt computing them also for stuf with no taginfos
    #Some useful BTAG vars
    process.patJets.userData.userFunctions = cms.vstring(
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").secondaryVertex(0).p4.M):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").secondaryVertex(0).numberOfSourceCandidatePtrs):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").flightDistance(0).value):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").flightDistance(0).significance):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").secondaryVertex(0).p4.x):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").secondaryVertex(0).p4.y):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").secondaryVertex(0).p4.z):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").secondaryVertex(0).vertex.x):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").secondaryVertex(0).vertex.y):(0)',
    '?(tagInfoCandSecondaryVertex("pfSecondaryVertex").nVertices()>0)?(tagInfoCandSecondaryVertex("pfSecondaryVertex").secondaryVertex(0).vertex.z):(0)',
    )
    process.patJets.userData.userFunctionLabels = cms.vstring('vtxMass','vtxNtracks','vtx3DVal','vtx3DSig','vtxPx','vtxPy','vtxPz','vtxPosX','vtxPosY','vtxPosZ')
    process.patJets.tagInfoSources = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos"))
    process.patJets.addTagInfos = cms.bool(True)
    #
    ## PU JetID
    process.load("RecoJets.JetProducers.PileupJetID_cfi")
    process.patJets.userData.userFloats.src = [ cms.InputTag("pileupJetId:fullDiscriminant"), ]

    ## CaloJets
    process.caloJetMap = cms.EDProducer("RecoJetDeltaRValueMapProducer",
         src = process.patJets.jetSource,
         matched = cms.InputTag("ak4CaloJets"),
         distMax = cms.double(0.4),
         values = cms.vstring('pt','emEnergyFraction'),
	 valueLabels = cms.vstring('pt','emEnergyFraction'),
	 lazyParser = cms.bool(True) )
    process.patJets.userData.userFloats.src += [ cms.InputTag("caloJetMap:pt"), cms.InputTag("caloJetMap:emEnergyFraction") ]

    #EGM object modifications
    from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import egamma_modifications
    process.slimmedElectrons.modifierConfig.modifications = egamma_modifications
    process.slimmedPhotons.modifierConfig.modifications   = egamma_modifications

    #VID Electron IDs
    electron_ids = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_PHYS14_PU20bx25_V2_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_50ns_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV60_cff',
                    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff']
    switchOnVIDElectronIdProducer(process,DataFormat.MiniAOD)
    process.egmGsfElectronIDs.physicsObjectSrc = \
        cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.electronMVAValueMapProducer.src = \
        cms.InputTag('reducedEgamma','reducedGedGsfElectrons')
    process.electronRegressionValueMapProducer.src = \
        cms.InputTag('reducedEgamma','reducedGedGsfElectrons')
    for idmod in electron_ids:
        setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection,None,False)

    #VID Photon IDs
    photon_ids = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_PHYS14_PU20bx25_V2_cff',
                  'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_50ns_V1_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V2_cff',
                  'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V2_cff']
    switchOnVIDPhotonIdProducer(process,DataFormat.MiniAOD) 
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
        setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection,None,False)
    
    # Adding puppi jets
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    process.load('RecoJets.JetProducers.ak4PFJetsPuppi_cfi')
    process.ak4PFJetsPuppi.doAreaFastjet = True # even for standard ak4PFJets this is overwritten in RecoJets/Configuration/python/RecoPFJets_cff
    #process.puppi.candName = cms.InputTag('packedPFCandidates')
    #process.puppi.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')
    # kind of ugly, is there a better way to do this?
    process.pfNoLepPUPPI = cms.EDFilter("PdgIdCandViewSelector",
        src = cms.InputTag("particleFlow"), 
        pdgId = cms.vint32( 1,2,22,111,130,310,2112,211,-211,321,-321,999211,2212,-2212 )
    )
    process.pfLeptonsPUPPET = cms.EDFilter("PdgIdCandViewSelector",
        src = cms.InputTag("particleFlow"),
        pdgId = cms.vint32(-11,11,-13,13),
    )
    process.puppiNoLep = process.puppi.clone()
    process.puppiNoLep.candName = cms.InputTag('pfNoLepPUPPI') 

    from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import j2tParametersVX
    process.ak4PFJetsPuppiTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
        j2tParametersVX,
        jets = cms.InputTag("ak4PFJetsPuppi")
    )
    process.patJetPuppiCharge = cms.EDProducer("JetChargeProducer",
        src = cms.InputTag("ak4PFJetsPuppiTracksAssociatorAtVertex"),
        var = cms.string('Pt'),
        exp = cms.double(1.0)
    )

    addJetCollection(process, postfix   = "", labelName = 'Puppi', jetSource = cms.InputTag('ak4PFJetsPuppi'),
                    jetCorrections = ('AK4PFchs', ['L2Relative', 'L3Absolute'], ''),
                    algo= 'AK', rParam = 0.4, btagDiscriminators = map(lambda x: x.value() ,process.patJets.discriminatorSources)
                    )
    
    process.patJetGenJetMatchPuppi.matched = 'slimmedGenJets'
    
    process.patJetsPuppi.userData.userFloats.src = cms.VInputTag(cms.InputTag(""))
    process.patJetsPuppi.jetChargeSource = cms.InputTag("patJetPuppiCharge")

    process.selectedPatJetsPuppi.cut = cms.string("pt > 20")

    process.load('PhysicsTools.PatAlgos.slimming.slimmedJets_cfi')
    process.slimmedJetsPuppi = process.slimmedJets.clone()
    process.slimmedJetsPuppi.src = cms.InputTag("selectedPatJetsPuppi")    
    process.slimmedJetsPuppi.packedPFCandidates = cms.InputTag("packedPFCandidates")

    ## puppi met
    process.load('RecoMET.METProducers.PFMET_cfi')
    process.puppiForMET = cms.EDProducer("CandViewMerger",
        src = cms.VInputTag( "pfLeptonsPUPPET", "puppiNoLep")
    ) 
    process.pfMetPuppi = process.pfMet.clone()
    process.pfMetPuppi.src = cms.InputTag("puppiForMET")
    process.pfMetPuppi.alias = cms.string('pfMetPuppi')
    # type1 correction, from puppi jets
    process.corrPfMetType1Puppi = process.corrPfMetType1.clone(
        src = 'ak4PFJetsPuppi',
        jetCorrLabel = 'ak4PFCHSL2L3Corrector',
    )
    del process.corrPfMetType1Puppi.offsetCorrLabel # no L1 for PUPPI jets
    process.pfMetT1Puppi = process.pfMetT1.clone(
        src = 'pfMetPuppi',
        srcCorrections = [ cms.InputTag("corrPfMetType1Puppi","type1") ]
    )

    from PhysicsTools.PatAlgos.tools.metTools import addMETCollection
    addMETCollection(process, labelName='patMETPuppi',   metSource='pfMetT1Puppi') # T1
    addMETCollection(process, labelName='patPFMetPuppi', metSource='pfMetPuppi')   # RAW

    process.load('PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi')
    process.slimmedMETsPuppi = process.slimmedMETs.clone()
    process.slimmedMETsPuppi.src = cms.InputTag("patMETPuppi")
    process.slimmedMETsPuppi.rawVariation   = cms.InputTag("patPFMetPuppi") # only central value
    # only central values for puppi met
    del process.slimmedMETsPuppi.t01Variation
    del process.slimmedMETsPuppi.t1SmearedVarsAndUncs
    del process.slimmedMETsPuppi.tXYUncForRaw
    del process.slimmedMETsPuppi.tXYUncForT1
    del process.slimmedMETsPuppi.tXYUncForT01
    del process.slimmedMETsPuppi.tXYUncForT1Smear
    del process.slimmedMETsPuppi.tXYUncForT01Smear
    del process.slimmedMETsPuppi.caloMET


def miniAOD_customizeMC(process):
    #slimmed pileup information
    process.load('PhysicsTools.PatAlgos.slimming.slimmedAddPileupInfo_cfi')
    
    process.muonMatch.matched = "prunedGenParticles"
    process.electronMatch.matched = "prunedGenParticles"
    process.electronMatch.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.photonMatch.matched = "prunedGenParticles"
    process.photonMatch.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.tauMatch.matched = "prunedGenParticles"
    process.tauGenJets.GenParticles = "prunedGenParticles"
    process.patJetPartons.particles = "prunedGenParticles"
    process.patJetPartonMatch.matched = "prunedGenParticles"
    process.patJetPartonMatch.mcStatus = [ 3, 23 ]
    process.patJetGenJetMatch.matched = "slimmedGenJets"
    process.patJetGenJetMatchAK8.matched =  "slimmedGenJetsAK8"
    process.patMuons.embedGenMatch = False
    process.patElectrons.embedGenMatch = False
    process.patPhotons.embedGenMatch = False
    process.patTaus.embedGenMatch = False
    process.patJets.embedGenPartonMatch = False
    #also jet flavour must be switched
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

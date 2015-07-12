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
    # apply type I/type I + II PFMEt corrections to pat::MET object
    # and estimate systematic uncertainties on MET
    # FIXME: are we 100% sure this should still be PF and not PFchs? 
    from PhysicsTools.PatUtils.tools.runType1PFMEtUncertainties import runType1PFMEtUncertainties
    addJetCollection(process, postfix   = "ForMetUnc", labelName = 'AK4PF', jetSource = cms.InputTag('ak4PFJets'), jetCorrections = ('AK4PF', ['L1FastJet', 'L2Relative', 'L3Absolute'], ''))
    process.patJetsAK4PFForMetUnc.getJetMCFlavour = False
    runType1PFMEtUncertainties(process,
                               addToPatDefaultSequence=False,
                               jetCollectionUnskimmed="patJetsAK4PFForMetUnc",
                               jetCollection="selectedPatJetsAK4PFForMetUnc",
                               electronCollection="selectedPatElectrons",
                               muonCollection="selectedPatMuons",
                               tauCollection="selectedPatTaus",
                               makeType1p2corrPFMEt=True,
                               doSmearJets=False,
                               outputModule=None)

    from PhysicsTools.PatAlgos.tools.metTools import addMETCollection
    addMETCollection(process,
                     labelName = "patCaloMet",
                     metSource = "caloMetM"
                     )
  

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
    process.load("PhysicsTools.PatAlgos.slimming.pileupJetId_cfi")
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
    electron_ids = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_CSA14_50ns_V1_cff',
                    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_CSA14_PU20bx25_V0_cff',
                    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV50_CSA14_25ns_cff',
                    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV50_CSA14_startup_cff']
    switchOnVIDElectronIdProducer(process, DataFormat.MiniAOD)
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
        setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection,None,False)
    
    # Adding puppi jets
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    process.load('RecoJets.JetProducers.ak4PFJetsPuppi_cfi')
    process.ak4PFJetsPuppi.doAreaFastjet = True # even for standard ak4PFJets this is overwritten in RecoJets/Configuration/python/RecoPFJets_cff
    #process.puppi.candName = cms.InputTag('packedPFCandidates')
    #process.puppi.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')
    
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
    process.pfMetPuppi = process.pfMet.clone()
    process.pfMetPuppi.src = cms.InputTag("puppi")
    process.pfMetPuppi.alias = cms.string('pfMetPuppi')
    ## type1 correction, from puppi jets
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
    process.slimmedMETsPuppi.rawUncertainties   = cms.InputTag("patPFMetPuppi") # only central value
    process.slimmedMETsPuppi.type1Uncertainties = cms.InputTag("patPFMetT1")    # only central value for now
    del process.slimmedMETsPuppi.type1p2Uncertainties # not available

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
    process.patJetGenJetMatchAK8.matched =  "slimmedGenJetsAK8"
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

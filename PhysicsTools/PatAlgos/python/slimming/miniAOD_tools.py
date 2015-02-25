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

    #add AK8
    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    addJetCollection(process, labelName = 'AK8',
                     jetSource = cms.InputTag('ak8PFJetsCHS'),
                     algo= 'AK', rParam = 0.8,
                     jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
                     btagInfos = ['caTopTagInfosPAT']
                     )
    process.patJetsAK8.userData.userFloats.src = [] # start with empty list of user floats
    process.selectedPatJetsAK8.cut = cms.string("pt > 100")
    process.patJetGenJetMatchAK8.matched =  'slimmedGenJets'
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

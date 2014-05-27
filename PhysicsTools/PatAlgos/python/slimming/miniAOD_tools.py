import FWCore.ParameterSet.Config as cms

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
    process.elPFIsoDepositCharged.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.elPFIsoDepositChargedAll.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.elPFIsoDepositNeutral.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.elPFIsoDepositGamma.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.elPFIsoDepositPU.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
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

    process.phPFIsoDepositCharged.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.phPFIsoDepositChargedAll.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.phPFIsoDepositNeutral.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.phPFIsoDepositGamma.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.phPFIsoDepositPU.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    #
    process.selectedPatJets.cut = cms.string("pt > 10")
    process.selectedPatMuons.cut = cms.string("pt > 5 || isPFMuon || (pt > 3 && (isGlobalMuon || isStandAloneMuon || numberOfMatches > 0 || muonID('RPCMuLoose')))") 
    process.selectedPatElectrons.cut = cms.string("") 
    process.selectedPatTaus.cut = cms.string("pt > 20 && tauID('decayModeFinding')> 0.5")
    process.selectedPatPhotons.cut = cms.string("")
    #
    from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection
    #switch to AK4 (though it should soon be unnecessary as ak4 should become the 71X default)
    #FIXME: still using AK5PFchs for jet energy corrections, while waiting for a new globalTag
    switchJetCollection(process, jetSource = cms.InputTag('ak4PFJetsCHS'),  
    jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), ''),
    btagDiscriminators = ['jetBProbabilityBJetTags', 'jetProbabilityBJetTags', 'trackCountingHighPurBJetTags', 'trackCountingHighEffBJetTags', 'simpleSecondaryVertexHighEffBJetTags',
                         'simpleSecondaryVertexHighPurBJetTags', 'combinedSecondaryVertexBJetTags' , 'combinedInclusiveSecondaryVertexBJetTags' ],
    )
    #add CA8   
    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    #
    addJetCollection(process, labelName = 'CA8', jetSource = cms.InputTag('ca8PFJetsCHS'),algo= 'CA', rParam = 0.8)
    process.selectedPatJetsCA8.cut = cms.string("pt > 100")
    process.patJetGenJetMatchCA8.matched =  'slimmedGenJets'
    #
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTriggerStandAlone
    switchOnTriggerStandAlone( process, outputModule = '' )
    process.patTrigger.packTriggerPathNames = cms.bool(True)
    #
    # apply type I/type I + II PFMEt corrections to pat::MET object
    # and estimate systematic uncertainties on MET
    # FIXME: this and the typeI MET should become AK4 once we have the proper JEC?
    from PhysicsTools.PatUtils.tools.metUncertaintyTools import runMEtUncertainties
    addJetCollection(process, postfix   = "ForMetUnc", labelName = 'AK5PF', jetSource = cms.InputTag('ak5PFJets'), jetCorrections = ('AK5PF', ['L1FastJet', 'L2Relative', 'L3Absolute'], ''))
    process.patJetsAK5PFForMetUnc.getJetMCFlavour = False
    runMEtUncertainties(process,jetCollection="selectedPatJetsAK5PFForMetUnc", outputModule=None)

    #keep this after all addJetCollections otherwise it will attempt computing them also for stuf with no taginfos
    #Some useful BTAG vars
    process.patJets.userData.userFunctions = cms.vstring(
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).p4.M):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().secondaryVertex(0).nTracks):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().flightDistance(0).value):(0)',
    '?(tagInfoSecondaryVertex().nVertices()>0)?(tagInfoSecondaryVertex().flightDistance(0).significance):(0)',
    )
    process.patJets.userData.userFunctionLabels = cms.vstring('vtxMass','vtxNtracks','vtx3DVal','vtx3DSig')
    process.patJets.tagInfoSources = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
    process.patJets.addTagInfos = cms.bool(True)
    #
    ## PU JetID
    process.load("PhysicsTools.PatAlgos.slimming.pileupJetId_cfi")
    process.patJets.userData.userFloats.src = [ cms.InputTag("pileupJetId:fullDiscriminant"), ]


def miniAOD_customizeMC(process):
    process.muonMatch.matched = "prunedGenParticles"
    process.electronMatch.matched = "prunedGenParticles"
    process.electronMatch.src = cms.InputTag("reducedEgamma","reducedGedGsfElectrons")
    process.photonMatch.matched = "prunedGenParticles"
    process.photonMatch.src = cms.InputTag("reducedEgamma","reducedGedPhotons")
    process.tauMatch.matched = "prunedGenParticles"
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

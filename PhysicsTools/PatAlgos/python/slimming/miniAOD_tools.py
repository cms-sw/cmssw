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
    process.patElectrons.embedPflowSuperCluster         = False
    process.patElectrons.embedPflowBasicClusters        = False
    process.patElectrons.embedPflowPreshowerClusters    = False
    #
    process.selectedPatJets.cut = cms.string("pt > 10")
    process.selectedPatMuons.cut = cms.string("pt > 5 || isPFMuon || (pt > 3 && (isGlobalMuon || isStandAloneMuon || numberOfMatches > 0 || muonID('RPCMuLoose')))") 
    process.selectedPatElectrons.cut = cms.string("") 
    process.selectedPatTaus.cut = cms.string("pt > 20 && tauID('decayModeFinding')> 0.5")
    process.selectedPatPhotons.cut = cms.string("pt > 15 && hadTowOverEm()<0.15 ")
    #
    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    #
    addJetCollection(process, labelName = 'CA8', jetSource = cms.InputTag('ca8PFJetsCHS') )
    process.selectedPatJetsCA8.cut = cms.string("pt > 100")
    process.patJetGenJetMatchPatJetsCA8.matched =  'slimmedGenJets'
    #
    ## PU JetID
    process.load("PhysicsTools.PatAlgos.slimming.pileupJetId_cfi")
    process.patJets.userData.userFloats.src = [ cms.InputTag("pileupJetId:fullDiscriminant"), ]
    #
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
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTriggerStandAlone
    switchOnTriggerStandAlone( process )
    process.patTrigger.packTriggerPathNames = cms.bool(True)
    #
    # apply type I/type I + II PFMEt corrections to pat::MET object
    # and estimate systematic uncertainties on MET
    from PhysicsTools.PatUtils.tools.metUncertaintyTools import runMEtUncertainties
    addJetCollection(process, postfix   = "ForMetUnc", labelName = 'AK5PF', jetSource = cms.InputTag('ak5PFJets'), jetCorrections = ('AK5PF', ['L1FastJet', 'L2Relative', 'L3Absolute'], ''), btagDiscriminators = ['combinedSecondaryVertexBJetTags' ] )
    runMEtUncertainties(process,jetCollection="selectedPatJetsAK5PFForMetUnc", outputModule=None)


def miniAOD_customizeMC(process):
    process.muonMatch.matched = "prunedGenParticles"
    process.electronMatch.matched = "prunedGenParticles"
    process.photonMatch.matched = "prunedGenParticles"
    process.tauMatch.matched = "prunedGenParticles"
    process.patJetPartonMatch.matched = "prunedGenParticles"
    process.patJetGenJetMatch.matched = "slimmedGenJets"
    process.patMuons.embedGenMatch = False
    process.patElectrons.embedGenMatch = False
    process.patPhotons.embedGenMatch = False
    process.patTaus.embedGenMatch = False
    process.patJets.embedGenPartonMatch = False

def miniAOD_customizeOutput(out):
    out.dropMetaData = cms.untracked.string('ALL')
    out.fastCloning= cms.untracked.bool(False)
    out.overrideInputFileSplitLevels = cms.untracked.bool(True)
    out.compressionAlgorithm = cms.untracked.string('LZMA')

def miniAOD_customizeData(process):
    from PhysicsTools.PatAlgos.tools.coreTools import runOnData
    runOnData( process )


import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers  import getPatAlgosToolsTask, addToProcessAndTask

def miniAODFromMiniAOD_customizeCommon(process):

    task = getPatAlgosToolsTask(process)

    ###########################################################################
    # Set puppi producers to use the original packedPFCandidate collection
    ###########################################################################
    process.packedpuppi.candName = 'packedPFCandidates::@skipCurrentProcess'
    process.packedpuppiNoLep.candName = 'packedPFCandidates::@skipCurrentProcess'

    ###########################################################################
    # Update packedPFCandidates with the recomputed puppi weights
    ###########################################################################
    addToProcessAndTask("packedPFCandidates", cms.EDProducer("PATPackedCandidateUpdater",
        src = cms.InputTag("packedPFCandidates", processName=cms.InputTag.skipCurrentProcess()),
        updatePuppiWeights = cms.bool(True),
        puppiWeight = cms.InputTag("packedpuppi"),
        puppiWeightNoLep = cms.InputTag("packedpuppiNoLep"),
      ),
      process, task
    )

    ###########################################################################
    # Recompute Puppi Isolation for muons, electrons and photons
    ###########################################################################
    from RecoMuon.MuonIsolation.muonIsolationPUPPI_cff import muonIsolationMiniAODPUPPI as _muonIsolationMiniAODPUPPI
    from RecoMuon.MuonIsolation.muonIsolationPUPPI_cff import muonIsolationMiniAODPUPPINoLeptons as _muonIsolationMiniAODPUPPINoLeptons

    addToProcessAndTask('muonPUPPIIsolation', _muonIsolationMiniAODPUPPI.clone(
        srcToIsolate = cms.InputTag('slimmedMuons', processName=cms.InputTag.skipCurrentProcess()),
        srcForIsolationCone = cms.InputTag('packedPFCandidates', processName=cms.InputTag.skipCurrentProcess()),
        puppiValueMap = cms.InputTag('packedpuppi'),
      ),
      process, task
    )

    addToProcessAndTask('muonPUPPINoLeptonsIsolation', _muonIsolationMiniAODPUPPINoLeptons.clone(
        srcToIsolate = cms.InputTag('slimmedMuons', processName=cms.InputTag.skipCurrentProcess()),
        srcForIsolationCone = cms.InputTag('packedPFCandidates', processName=cms.InputTag.skipCurrentProcess()),
        puppiValueMap = cms.InputTag('packedpuppiNoLep'),
      ),
      process, task
    )

    addToProcessAndTask('slimmedMuonsUpdatedPuppiIsolation', cms.EDProducer('PATMuonPuppiIsolationUpdater',
        src = cms.InputTag('slimmedMuons', processName=cms.InputTag.skipCurrentProcess()),
        puppiIsolationChargedHadrons          = cms.InputTag("muonPUPPIIsolation","h+-DR040-ThresholdVeto000-ConeVeto000"),
        puppiIsolationNeutralHadrons          = cms.InputTag("muonPUPPIIsolation","h0-DR040-ThresholdVeto000-ConeVeto001"),
        puppiIsolationPhotons                 = cms.InputTag("muonPUPPIIsolation","gamma-DR040-ThresholdVeto000-ConeVeto001"),
        puppiNoLeptonsIsolationChargedHadrons = cms.InputTag("muonPUPPINoLeptonsIsolation","h+-DR040-ThresholdVeto000-ConeVeto000"),
        puppiNoLeptonsIsolationNeutralHadrons = cms.InputTag("muonPUPPINoLeptonsIsolation","h0-DR040-ThresholdVeto000-ConeVeto001"),
        puppiNoLeptonsIsolationPhotons        = cms.InputTag("muonPUPPINoLeptonsIsolation","gamma-DR040-ThresholdVeto000-ConeVeto001"),
      ),
      process, task
    )

    from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationPUPPI_cff import egmPhotonIsolationMiniAODPUPPI as _egmPhotonPUPPIIsolationForPhotons
    from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPI as _egmElectronIsolationMiniAODPUPPI
    from RecoEgamma.EgammaIsolationAlgos.egmElectronIsolationPUPPI_cff import egmElectronIsolationMiniAODPUPPINoLeptons as _egmElectronIsolationMiniAODPUPPINoLeptons

    addToProcessAndTask('egmPhotonPUPPIIsolation', _egmPhotonPUPPIIsolationForPhotons.clone(
        srcToIsolate = cms.InputTag('slimmedPhotons', processName=cms.InputTag.skipCurrentProcess()),
        srcForIsolationCone = cms.InputTag("packedPFCandidates", processName=cms.InputTag.skipCurrentProcess()),
        puppiValueMap = cms.InputTag('packedpuppi'),
      ),
      process, task
    )

    addToProcessAndTask('egmElectronPUPPIIsolation', _egmElectronIsolationMiniAODPUPPI.clone(
        srcToIsolate = cms.InputTag('slimmedElectrons', processName=cms.InputTag.skipCurrentProcess()),
        srcForIsolationCone = cms.InputTag("packedPFCandidates", processName=cms.InputTag.skipCurrentProcess()),
        puppiValueMap = cms.InputTag('packedpuppi'),
      ),
      process, task
    )

    addToProcessAndTask('egmElectronPUPPINoLeptonsIsolation', _egmElectronIsolationMiniAODPUPPI.clone(
        srcToIsolate = cms.InputTag('slimmedElectrons', processName=cms.InputTag.skipCurrentProcess()),
        srcForIsolationCone = cms.InputTag("packedPFCandidates", processName=cms.InputTag.skipCurrentProcess()),
        puppiValueMap = cms.InputTag('packedpuppiNoLep'),
      ),
      process, task
    )

    addToProcessAndTask('slimmedPhotonsUpdatedPuppiIsolation', cms.EDProducer('PATPhotonPuppiIsolationUpdater',
        src = cms.InputTag('slimmedPhotons', processName=cms.InputTag.skipCurrentProcess()),
        puppiIsolationChargedHadrons = cms.InputTag("egmPhotonPUPPIIsolation","h+-DR030-"),
        puppiIsolationNeutralHadrons = cms.InputTag("egmPhotonPUPPIIsolation","h0-DR030-"),
        puppiIsolationPhotons        = cms.InputTag("egmPhotonPUPPIIsolation","gamma-DR030-"),
      ),
      process, task
    )

    addToProcessAndTask('slimmedElectronsUpdatedPuppiIsolation', cms.EDProducer('PATElectronPuppiIsolationUpdater',
        src = cms.InputTag('slimmedElectrons', processName=cms.InputTag.skipCurrentProcess()),
        puppiIsolationChargedHadrons          = cms.InputTag("egmElectronPUPPIIsolation","h+-DR030-BarVeto000-EndVeto001"),
        puppiIsolationNeutralHadrons          = cms.InputTag("egmElectronPUPPIIsolation","h0-DR030-BarVeto000-EndVeto000"),
        puppiIsolationPhotons                 = cms.InputTag("egmElectronPUPPIIsolation","gamma-DR030-BarVeto000-EndVeto008"),
        puppiNoLeptonsIsolationChargedHadrons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","h+-DR030-BarVeto000-EndVeto001"),
        puppiNoLeptonsIsolationNeutralHadrons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","h0-DR030-BarVeto000-EndVeto000"),
        puppiNoLeptonsIsolationPhotons        = cms.InputTag("egmElectronPUPPINoLeptonsIsolation","gamma-DR030-BarVeto000-EndVeto008"),
      ),
      process, task
    )

    ###########################################################################
    # Rekey jet constituents
    ###########################################################################
    addToProcessAndTask("slimmedJets", cms.EDProducer("PATJetCandidatesRekeyer",
        src = cms.InputTag("slimmedJets", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates",processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("slimmedJetsPuppiPreRekey", process.slimmedJetsPuppi.clone(), process, task)
    del process.slimmedJetsPuppi

    addToProcessAndTask("slimmedJetsPuppi", cms.EDProducer("PATJetCandidatesRekeyer",
        src = cms.InputTag("slimmedJetsPuppiPreRekey"),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates",processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("slimmedJetsAK8PreRekey", process.slimmedJetsAK8.clone(), process, task)
    del process.slimmedJetsAK8

    addToProcessAndTask("slimmedJetsAK8", cms.EDProducer("PATJetCandidatesRekeyer",
        src = cms.InputTag("slimmedJetsAK8PreRekey"),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates",processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("selectedUpdatedPatJetsAK8PFPuppiSoftDropSubjetsSlimmedDeepFlavourPreRekey",
        process.selectedUpdatedPatJetsAK8PFPuppiSoftDropSubjetsSlimmedDeepFlavour.clone(),
        process,
        task
    )
    del process.selectedUpdatedPatJetsAK8PFPuppiSoftDropSubjetsSlimmedDeepFlavour

    addToProcessAndTask("selectedUpdatedPatJetsAK8PFPuppiSoftDropSubjetsSlimmedDeepFlavour", cms.EDProducer("PATJetCandidatesRekeyer",
        src = cms.InputTag("selectedUpdatedPatJetsAK8PFPuppiSoftDropSubjetsSlimmedDeepFlavourPreRekey"),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates",processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    ###########################################################################
    # Rekey tau constituents
    ###########################################################################
    addToProcessAndTask("slimmedTaus", cms.EDProducer("PATTauCandidatesRekeyer",
        src = cms.InputTag("slimmedTaus", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates",processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("slimmedTausBoosted", cms.EDProducer("PATTauCandidatesRekeyer",
        src = cms.InputTag("slimmedTausBoosted", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates",processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    ###########################################################################
    # Rekey candidates in electrons, photons and muons
    ###########################################################################
    addToProcessAndTask("slimmedElectrons", cms.EDProducer("PATElectronCandidatesRekeyer",
        src = cms.InputTag("slimmedElectronsUpdatedPuppiIsolation"),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates", processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("slimmedLowPtElectrons", cms.EDProducer("PATElectronCandidatesRekeyer",
        src = cms.InputTag("slimmedLowPtElectrons", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates", processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("slimmedPhotons", cms.EDProducer("PATPhotonCandidatesRekeyer",
        src = cms.InputTag("slimmedPhotonsUpdatedPuppiIsolation"),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates", processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )


    addToProcessAndTask("slimmedMuons", cms.EDProducer("PATMuonCandidatesRekeyer",
        src = cms.InputTag("slimmedMuonsUpdatedPuppiIsolation"),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates", processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("slimmedDisplacedMuons", cms.EDProducer("PATMuonCandidatesRekeyer",
        src = cms.InputTag("slimmedDisplacedMuons", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates", processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    ###########################################################################
    # Rekey daughters of secondary vertices
    ###########################################################################
    addToProcessAndTask("slimmedKshortVertices", cms.EDProducer("VertexCompositeCandidateDaughtersRekeyer",
        src = cms.InputTag("slimmedKshortVertices", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesOri = cms.InputTag("packedPFCandidates", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates", processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("slimmedLambdaVertices", cms.EDProducer("VertexCompositeCandidateDaughtersRekeyer",
        src = cms.InputTag("slimmedLambdaVertices", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesOri = cms.InputTag("packedPFCandidates", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates", processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    addToProcessAndTask("slimmedSecondaryVertices", cms.EDProducer("VertexCompositeCandidateDaughtersRekeyer",
        src = cms.InputTag("slimmedSecondaryVertices", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesOri = cms.InputTag("packedPFCandidates", processName=cms.InputTag.skipCurrentProcess()),
        packedPFCandidatesNew = cms.InputTag("packedPFCandidates", processName=cms.InputTag.currentProcess()),
      ),
      process, task
    )

    mini_output = None
    for out_name in process.outputModules_().keys():
        if out_name.startswith('MINIAOD'):
            mini_output = getattr(process, out_name)
            break
    if mini_output:
        for new_collection_to_keep in ['packedPFCandidates',
                                       'slimmedJets',
                                       'slimmedJetsPuppi',
                                       'slimmedJetsAK8',
                                       'slimmedJetsAK8PFPuppiSoftDropPacked_SubJets',
                                       'slimmedMETsPuppi',
                                       'slimmedTaus',
                                       'slimmedTausBoosted',
                                       'slimmedElectrons',
                                       'slimmedMuons',
                                       'slimmedPhotons',
                                       'slimmedLowPtElectrons',
                                       'slimmedKshortVertices',
                                       'slimmedLambdaVertices',
                                       'slimmedSecondaryVertices']:
            new_collection_to_keep += '_*' if not '_' in new_collection_to_keep else ''
            mini_output.outputCommands += [
                f'drop *_{new_collection_to_keep}_*',
                f'keep *_{new_collection_to_keep}_{process.name_()}']

    return process

def miniAODFromMiniAOD_customizeData(process):
    from PhysicsTools.PatAlgos.tools.puppiJetMETReclusteringFromMiniAOD_cff import puppiJetMETReclusterFromMiniAOD_Data
    process = puppiJetMETReclusterFromMiniAOD_Data(process)
    return process

def miniAODFromMiniAOD_customizeMC(process):
    from PhysicsTools.PatAlgos.tools.puppiJetMETReclusteringFromMiniAOD_cff import puppiJetMETReclusterFromMiniAOD_MC
    process = puppiJetMETReclusterFromMiniAOD_MC(process)
    return process

def miniAODFromMiniAOD_customizeAllData(process):
    process = miniAODFromMiniAOD_customizeData(process)
    process = miniAODFromMiniAOD_customizeCommon(process)
    return process

def miniAODFromMiniAOD_customizeAllMC(process):
    process = miniAODFromMiniAOD_customizeMC(process)
    process = miniAODFromMiniAOD_customizeCommon(process)
    return process

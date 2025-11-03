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

    # Add deep flavour info to AK4CHS jets
    # (based on function from jetsAK4_CHS_cff in Nano)
    from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
    def _m2m_addDeepInfoAK4CHS(process,
                               addDeepBTag,
                               addDeepFlavour,
                               addParticleNet,
                               addRobustParTAK4=False,
                               addUnifiedParTAK4=False):
        _btagDiscriminators=[]
        if addDeepBTag:
            print("Updating process to run DeepCSV btag")
            _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb','pfDeepCSVJetTags:probc']
        if addDeepFlavour:
            print("Updating process to run DeepFlavour btag")
            _btagDiscriminators += ['pfDeepFlavourJetTags:probb','pfDeepFlavourJetTags:probbb','pfDeepFlavourJetTags:problepb','pfDeepFlavourJetTags:probc']
        if addParticleNet:
            print("Updating process to run ParticleNetAK4")
            from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll as pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll
            from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll as pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll
            _btagDiscriminators += pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll
            _btagDiscriminators += pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll
        if addRobustParTAK4:
            print("Updating process to run RobustParTAK4")
            from RecoBTag.ONNXRuntime.pfParticleTransformerAK4_cff import _pfParticleTransformerAK4JetTagsAll as pfParticleTransformerAK4JetTagsAll
            _btagDiscriminators += pfParticleTransformerAK4JetTagsAll
        if addUnifiedParTAK4:
            print("Updating process to run UnifiedParTAK4")
            from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4_cff import _pfUnifiedParticleTransformerAK4JetTagsAll as pfUnifiedParticleTransformerAK4JetTagsAll
            _btagDiscriminators += pfUnifiedParticleTransformerAK4JetTagsAll

        if len(_btagDiscriminators)==0: return process
        print("Will recalculate the following discriminators: "+", ".join(_btagDiscriminators))
        updateJetCollection(
            process,
            jetSource = cms.InputTag('slimmedJets', processName=cms.InputTag.skipCurrentProcess()),
            jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute','L2L3Residual']), 'None'),
            btagDiscriminators = _btagDiscriminators,
            postfix = 'AK4CHSWithDeepInfo',
        )
        process.load("Configuration.StandardSequences.MagneticField_cff")

        from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask
        task = getPatAlgosToolsTask(process)
        addToProcessAndTask("slimmedJetsWithDeepInfo", process.selectedUpdatedPatJetsAK4CHSWithDeepInfo.clone(), process, task)
        del process.selectedUpdatedPatJetsAK4CHSWithDeepInfo

        return process

    # FIXME: this probably needs to be era dependent, but all enabled for now
    m2m_addDeepInfoAK4CHS_switch = cms.PSet(
        m2m_addDeepBTag_switch = cms.untracked.bool(True),
        m2m_addDeepFlavourTag_switch = cms.untracked.bool(True),
        m2m_addParticleNet_switch = cms.untracked.bool(True),
        m2m_addRobustParTAK4Tag_switch = cms.untracked.bool(True),
        m2m_addUnifiedParTAK4Tag_switch = cms.untracked.bool(True)
    )
    _m2m_addDeepInfoAK4CHS(process,
                           addDeepBTag = m2m_addDeepInfoAK4CHS_switch.m2m_addDeepBTag_switch,
                           addDeepFlavour = m2m_addDeepInfoAK4CHS_switch.m2m_addDeepFlavourTag_switch,
                           addParticleNet = m2m_addDeepInfoAK4CHS_switch.m2m_addParticleNet_switch,
                           addRobustParTAK4 = m2m_addDeepInfoAK4CHS_switch.m2m_addRobustParTAK4Tag_switch,
                           addUnifiedParTAK4 = m2m_addDeepInfoAK4CHS_switch.m2m_addUnifiedParTAK4Tag_switch)

    jets_ak4chs_to_use = 'slimmedJets::@skipCurrentProcess'
    for flag in m2m_addDeepInfoAK4CHS_switch._Parameterizable__parameterNames:
        if flag:
            jets_ak4chs_to_use = 'slimmedJetsWithDeepInfo'
            break

    addToProcessAndTask("slimmedJets", cms.EDProducer("PATJetCandidatesRekeyer",
        src = cms.InputTag(jets_ak4chs_to_use),
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
    from Configuration.ProcessModifiers.run2_miniAOD_miniAODUL_cff import run2_miniAOD_miniAODUL
    from Configuration.ProcessModifiers.run3_miniAOD_miniAODpre142X_cff import run3_miniAOD_miniAODpre142X
    import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
    taus_to_use = 'slimmedTaus::@skipCurrentProcess'
    idsToRun_for_taus = cms.PSet(idsToAdd=cms.vstring())
    run2_miniAOD_miniAODUL.toModify(
        idsToRun_for_taus, idsToAdd=["deepTau2018v2p5"])
    if idsToRun_for_taus.idsToAdd:
        if not hasattr(process,'tauTask'): process.tauTask = cms.Task()
        taus_from_mini = taus_to_use
        taus_to_use = 'slimmedTausUpdated'
        tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, debug=False,
                                                  originalTauName=taus_from_mini,
                                                  updatedTauName=taus_to_use,
                                                  postfix="M2M",
                                                  toKeep=idsToRun_for_taus.idsToAdd)
        tauIdEmbedder.runTauID()
        task.add(process.rerunMvaIsolationTaskM2M, getattr(process, taus_to_use))

    # Produce a "hybrid" tau collection combining HPS-reconstructed taus with tau-tagged jets
    def _m2m_addUTagToTaus(process, task,
                           tausToUpdate = 'slimmedTaus',
                           storePNetCHSjets = False,
                           storeUParTPUPPIjets = False,
                           addGenJet = False):
        if not (storePNetCHSjets or storeUParTPUPPIjets): return process
        noUpdatedTauName = f'{tausToUpdate}NoUTag'
        updatedTauName = ''
        addToProcessAndTask(noUpdatedTauName, getattr(process, tausToUpdate).clone(), process, task)
        delattr(process, tausToUpdate)
        if addGenJet:
            from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets
            addToProcessAndTask('tauGenJets',
                                tauGenJets.clone(GenParticles =  'prunedGenParticles'),
                                process, task)
            from PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi import tauGenJetsSelectorAllHadrons
            addToProcessAndTask('tauGenJetsSelectorAllHadrons',
                                tauGenJetsSelectorAllHadrons,
                                process, task)
            from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import tauGenJetMatch
        from PhysicsTools.PatAlgos.patTauHybridProducer_cfi import patTauHybridProducer
        if storePNetCHSjets:
            jetCollection = jets_ak4chs_to_use
            TagName = 'pfParticleNetFromMiniAODAK4CHSCentralJetTags'
            tag_prefix = 'byUTagCHS'
            updatedTauName = f'{tausToUpdate}WithUTagCHS'
            # PNet tagger used for CHS jets
            from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import pfParticleNetFromMiniAODAK4CHSCentralJetTags
            Discriminators = [TagName+":"+tag for tag in pfParticleNetFromMiniAODAK4CHSCentralJetTags.flav_names.value()]
            # Define "hybridTau" producer
            setattr(process, updatedTauName,
                    patTauHybridProducer.clone(
                        src = noUpdatedTauName,
                        jetSource = jetCollection,
                        dRMax = 0.4,
                        jetPtMin = 15,
                        jetEtaMax = 2.5,
                        UTagLabel = TagName,
                        UTagScoreNames = Discriminators,
                        tagPrefix = tag_prefix,
                        tauScoreMin = -1,
                        vsJetMin = 0.05,
                        checkTauScoreIsBest = False,
                        chargeAssignmentProbMin = 0.2,
                        addGenJetMatch = addGenJet,
                        genJetMatch = ''
                    ))
            if addGenJet:
                addToProcessAndTask('tauGenJetMatchCHSJet',
                                    tauGenJetMatch.clone(src = jetCollection),
                                    process, task)
                getattr(process,updatedTauName).genJetMatch = 'tauGenJetMatchCHSJet'
            if storeUParTPUPPIjets: task.add(getattr(process,updatedTauName))
        if storeUParTPUPPIjets:
            jetCollection = 'slimmedJetsPuppiPreRekey'
            TagName = 'pfUnifiedParticleTransformerAK4JetTags'
            tag_prefix = 'byUTagPUPPI'
            updatedTauName = f'{tausToUpdate}WithUTagPUPPI' if not storePNetCHSjets else f'{tausToUpdate}WithUTagCHSAndUTagPUPPI'
            # Unified ParT Tagger used for PUPPI jets
            from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4JetTags_cfi import pfUnifiedParticleTransformerAK4JetTags
            Discriminators = [TagName+":"+tag for tag in pfUnifiedParticleTransformerAK4JetTags.flav_names.value()]
            # Define "hybridTau" producer
            setattr(process, updatedTauName,
                    patTauHybridProducer.clone(
                        src = noUpdatedTauName if not storePNetCHSjets else f'{tausToUpdate}WithUTagCHS',
                        jetSource = jetCollection,
                        dRMax = 0.4,
                        jetPtMin = 15,
                        jetEtaMax = 2.5,
                        UTagLabel = TagName,
                        UTagScoreNames = Discriminators,
                        tagPrefix = tag_prefix,
                        tauScoreMin = -1,
                        vsJetMin = 0.05,
                        checkTauScoreIsBest = False,
                        chargeAssignmentProbMin = 0.2,
                        addGenJetMatch = addGenJet,
                        genJetMatch = ''
                    ))
            if addGenJet:
                addToProcessAndTask('tauGenJetMatchPUPPIJet',
                                    tauGenJetMatch.clone(src = jetCollection),
                                    process, task)
                getattr(process,updatedTauName).genJetMatch = 'tauGenJetMatchPUPPIJet'
        # Add "hybridTau" producer to pat-task
        addToProcessAndTask(tausToUpdate,
                            getattr(process, updatedTauName).clone(),
                            process, task)
        return process

    m2m_uTagToTaus_switches = cms.PSet(
        storePNetCHSjets = cms.bool(True),
        storeUParTPUPPIjets = cms.bool(True),
        addGenJet = cms.bool(True)
    )
    _m2m_addUTagToTaus(process, task,
                       taus_to_use,
                       storePNetCHSjets = m2m_uTagToTaus_switches.storePNetCHSjets.value(),
                       storeUParTPUPPIjets = m2m_uTagToTaus_switches.storeUParTPUPPIjets.value(),
                       addGenJet = m2m_uTagToTaus_switches.addGenJet.value()
    )

    addToProcessAndTask("slimmedTaus", cms.EDProducer("PATTauCandidatesRekeyer",
                                                      src = cms.InputTag(taus_to_use),
                                                      packedPFCandidatesNew = cms.InputTag("packedPFCandidates",processName=cms.InputTag.currentProcess()),
                                                      ),
                        process, task
                        )
    # fix circular module dependency in ParticleNetFromMiniAOD TagInfos when slimmedTaus is updated
    def _fixPNetInputCollection(process):
        if hasattr(process, 'slimmedTaus'):
            for mod in process.producers.keys():
                if 'ParticleNetFromMiniAOD' in mod and 'TagInfos' in mod:
                    getattr(process, mod).taus = 'slimmedTaus::@skipCurrentProcess'
    _fixPNetInputCollection(process)

    boosted_taus_to_use = 'slimmedTausBoosted::@skipCurrentProcess'
    idsToRun_for_boosted_taus = cms.PSet(idsToAdd=cms.vstring())
    run2_miniAOD_miniAODUL.toModify(
        idsToRun_for_boosted_taus, idsToAdd=["mvaIso", "mvaIsoNewDM", "mvaIsoDR0p3", "againstEle", "boostedDeepTauRunIIv2p0"])
    run3_miniAOD_miniAODpre142X.toModify(
        idsToRun_for_boosted_taus, idsToAdd=["boostedDeepTauRunIIv2p0"])
    if idsToRun_for_boosted_taus.idsToAdd:
        if not hasattr(process,'tauTask'): process.tauTask = cms.Task()
        boosted_taus_from_mini = boosted_taus_to_use
        boosted_taus_to_use = 'slimmedTausBoostedUpdated'
        tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, debug=False,
                                                  originalTauName=boosted_taus_from_mini,
                                                  updatedTauName=boosted_taus_to_use,
                                                  postfix="BoostedM2M",
                                                  toKeep=idsToRun_for_boosted_taus.idsToAdd)
        tauIdEmbedder.runTauID()
        task.add(process.rerunMvaIsolationTaskBoostedM2M, getattr(process, boosted_taus_to_use))
        
    addToProcessAndTask("slimmedTausBoosted", cms.EDProducer("PATTauCandidatesRekeyer",
                                                             src = cms.InputTag(boosted_taus_to_use),
                                                             packedPFCandidatesNew = cms.InputTag("packedPFCandidates",processName=cms.InputTag.currentProcess()),
                                                             ),
                        process, task
                        )

    # Use original pf-particles w/o recalculated puppi to be coherent with on-the-fly deepTau calculations
    for mod in process.producers.keys():
        if 'deeptau' in mod.lower():
            getattr(process, mod).pfcands = 'packedPFCandidates::@skipCurrentProcess'

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


    _modified_run2_task = task.copyAndExclude([getattr(process,thisone) for thisone in ['slimmedDisplacedMuons']])
    from PhysicsTools.PatAlgos.patRefitVertexProducer_cfi import patRefitVertexProducer
    process.offlineSlimmedPrimaryVerticesWithBS = patRefitVertexProducer.clone(
        srcVertices = "offlineSlimmedPrimaryVertices",
    )
    
    _modified_run2_task.add( process.offlineSlimmedPrimaryVerticesWithBS )
    run2_miniAOD_miniAODUL.toReplaceWith(
        task, _modified_run2_task)

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

def hybridTausOnData(process):
    print('removing MC dependencies for hybrid taus')
    modulesToDel = []
    for moduleName in process.producers.keys():
        if not 'slimmedTaus' in moduleName: continue
        module = getattr(process, moduleName)
        if module.__dict__['_TypedParameterizable__type'] != 'PATTauHybridProducer': continue
        module.addGenJetMatch = False
        genJetModule = module.genJetMatch.getModuleLabel()
        if not genJetModule in modulesToDel:
            modulesToDel.append(genJetModule)
        module.genJetMatch = ''
    if modulesToDel:
        modulesToDel.extend(['tauGenJets','tauGenJetsSelectorAllHadrons'])
    for moduleName in modulesToDel:
        if hasattr(process,moduleName): delattr(process,moduleName)
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
    # remove MC dependencies for hybrid taus
    process = hybridTausOnData(process)
    return process

def miniAODFromMiniAOD_customizeAllMC(process):
    process = miniAODFromMiniAOD_customizeMC(process)
    process = miniAODFromMiniAOD_customizeCommon(process)
    return process

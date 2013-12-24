#
#  SUSY-PAT configuration fragment
#
#  PAT configuration for the SUSY group - 53X series
#  More information here:
#  https://twiki.cern.ch/twiki/bin/view/CMS/SusyPatLayer1DefV12
#test


import FWCore.ParameterSet.Config as cms

def addDefaultSUSYPAT(process,mcInfo=True,HLTMenu='HLT',jetMetCorrections=['L2Relative', 'L3Absolute'],mcVersion='',theJetNames = ['AK5PF'],doValidation=False,extMatch=False,doSusyTopProjection=False,doType1MetCorrection=True,doType0MetCorrection=False):
    loadPF2PAT(process,mcInfo,jetMetCorrections,extMatch,doSusyTopProjection,doType1MetCorrection,doType0MetCorrection,'PF')
    addTagInfos(process,jetMetCorrections)
    if not mcInfo:
        removeMCDependence(process)
    loadPAT(process,jetMetCorrections,extMatch)
    addJetMET(process,theJetNames,jetMetCorrections,mcVersion)
    # loadType1METSequence(process)   # defines process.Type1METSequence
    # loadPATTriggers(process,HLTMenu,theJetNames,electronMatches,muonMatches,tauMatches,jetMatches,photonMatches)

    #-- Counter for the number of processed events --------------------------------
    process.eventCountProducer = cms.EDProducer("EventCountProducer")

    # Full path
    #process.load('RecoTauTag.Configuration.RecoPFTauTag_cff')
    process.susyPatDefaultSequence = cms.Sequence( process.eventCountProducer
                                                   # * process.PFTau
                                                   # * process.Type1METSequence
                                                   * process.patPF2PATSequence
                                                   * process.patPF2PATSequencePF
                                                   )

    if mcInfo and extMatch:
        extensiveMatching(process)
        process.susyPatDefaultSequence.replace(process.patDefaultSequence, process.extensiveMatching+process.patDefaultSequence)
    
    if doValidation:
        loadSusyValidation(process)
        process.susyPatDefaultSequence.replace(process.patPF2PATSequencePF, process.patPF2PATSequencePF * process.ak5CaloJetsL2L3 * process.metJESCorAK5CaloJet  * process.RecoSusyValidation * process.PatSusyValidation*process.MEtoEDMConverter)

def extensiveMatching(process):
    process.load("SimGeneral.TrackingAnalysis.trackingParticlesNoSimHits_cfi")    # On RECO
    process.load("SimMuon.MCTruth.MuonAssociatorByHits_cfi")  # On RECO
    process.mergedTruth = cms.EDProducer("GenPlusSimParticleProducer",
        src           = cms.InputTag("g4SimHits"), # use "famosSimHits" for FAMOS
        setStatus     = cms.int32(5),             # set status = 8 for GEANT GPs
        filter        = cms.vstring("pt > 0.0"),  # just for testing (optional)
        genParticles   = cms.InputTag("genParticles") # original genParticle list
    )
    process.load("MuonAnalysis.MuonAssociators.muonClassificationByHits_cfi")

    from MuonAnalysis.MuonAssociators.muonClassificationByHits_cfi import addUserData as addClassByHits
    addClassByHits(process.patMuons,labels=['classByHitsGlb'],extraInfo=True)
    addClassByHits(process.patMuonsPF,labels=['classByHitsGlb'],extraInfo=True)
    
    process.extensiveMatching = cms.Sequence(process.mergedTruth+process.muonClassificationByHits)

def loadPAT(process,jetMetCorrections,extMatch):
    #-- Changes for electron and photon ID ----------------------------------------
    from PhysicsTools.PatAlgos.tools.pfTools import usePFIso
    usePFIso( process )
    process.patElectrons.isolationValues = cms.PSet(
        pfNeutralHadrons = cms.InputTag("elPFIsoValueNeutral03PFIdPFIso"),
        pfChargedAll = cms.InputTag("elPFIsoValueChargedAll03PFIdPFIso"),
        pfPUChargedHadrons = cms.InputTag("elPFIsoValuePU03PFIdPFIso"),
        pfPhotons = cms.InputTag("elPFIsoValueGamma03PFIdPFIso"),
        pfChargedHadrons = cms.InputTag("elPFIsoValueCharged03PFIdPFIso")
        )
    process.patElectrons.isolationValuesNoPFId = cms.PSet(
        pfNeutralHadrons = cms.InputTag("elPFIsoValueNeutral03NoPFIdPFIso"),
        pfChargedAll = cms.InputTag("elPFIsoValueChargedAll03NoPFIdPFIso"),
        pfPUChargedHadrons = cms.InputTag("elPFIsoValuePU03NoPFIdPFIso"),
        pfPhotons = cms.InputTag("elPFIsoValueGamma03NoPFIdPFIso"),
        pfChargedHadrons = cms.InputTag("elPFIsoValueCharged03NoPFIdPFIso")
        )
    process.patDefaultSequence.replace(process.patElectrons,process.eleIsoSequence+process.patElectrons)
    process.patDefaultSequence.replace(process.patMuons,process.muIsoSequence+process.patMuons)
    
    # Turn off photon-electron cleaning (i.e., flag only)
    process.cleanPatPhotons.checkOverlaps.electrons.requireNoOverlaps = False

    # Embed tracks, since we drop them
    process.patElectrons.embedTrack = True
    process.patMuons.embedTrack   = True

    #include tau decay mode in pat::Taus (elese it will just be uninitialized)
    #decay modes are dropped and have to be redone, this is a bit dangorous since the decay modes insered are *not* the ones used in RECO
    #process.patTaus.addDecayMode = True
    #process.makePatTaus.replace( process.patTaus, process.shrinkingConePFTauDecayModeProducer + process.patTaus )

    #Additional electron ids as defined for VBTF
    process.load("ElectroWeakAnalysis.WENu.simpleEleIdSequence_cff")
    process.patElectrons.electronIDSources = cms.PSet(
    eidTight = cms.InputTag("eidTight"),
    eidLoose = cms.InputTag("eidLoose"),
    eidRobustTight = cms.InputTag("eidRobustTight"),
    eidRobustHighEnergy = cms.InputTag("eidRobustHighEnergy"),
    eidRobustLoose = cms.InputTag("eidRobustLoose"),
    simpleEleId95relIso= cms.InputTag("simpleEleId95relIso"),
    simpleEleId90relIso= cms.InputTag("simpleEleId90relIso"),
    simpleEleId85relIso= cms.InputTag("simpleEleId85relIso"),
    simpleEleId80relIso= cms.InputTag("simpleEleId80relIso"),
    simpleEleId70relIso= cms.InputTag("simpleEleId70relIso"),
    simpleEleId60relIso= cms.InputTag("simpleEleId60relIso"),
    simpleEleId95cIso= cms.InputTag("simpleEleId95cIso"),
    simpleEleId90cIso= cms.InputTag("simpleEleId90cIso"),
    simpleEleId85cIso= cms.InputTag("simpleEleId85cIso"),
    simpleEleId80cIso= cms.InputTag("simpleEleId80cIso"),
    simpleEleId70cIso= cms.InputTag("simpleEleId70cIso"),
    simpleEleId60cIso= cms.InputTag("simpleEleId60cIso"))
    process.patDefaultSequence.replace(process.patElectrons,process.simpleEleIdSequence+process.patElectrons)
    
    #-- Tuning of Monte Carlo matching --------------------------------------------
    # Also match with leptons of opposite charge
    process.electronMatch.checkCharge = False
    process.electronMatch.maxDeltaR   = cms.double(0.2)
    process.electronMatch.maxDPtRel   = cms.double(999999.)
    process.muonMatch.checkCharge     = False
    process.muonMatch.maxDeltaR       = cms.double(0.2)
    process.muonMatch.maxDPtRel       = cms.double(999999.)
    #process.tauMatch.checkCharge      = False
    #process.tauMatch.maxDeltaR        = cms.double(0.3)
    process.patJetPartonMatch.maxDeltaR  = cms.double(0.25)
    process.patJetPartonMatch.maxDPtRel  = cms.double(999999.)
    process.patJetGenJetMatch.maxDeltaR  = cms.double(0.25)
    process.patJetGenJetMatch.maxDPtRel  = cms.double(999999.)
    if extMatch:
        process.electronMatch.mcStatus = cms.vint32(1,5)
        process.electronMatch.matched = "mergedTruth"
        process.muonMatch.mcStatus = cms.vint32(1,5)
        process.muonMatch.matched = "mergedTruth"
        process.patJetPartonMatch.matched = "mergedTruth"
        process.patJetPartons.src = "mergedTruth"
        process.photonMatch.matched = "mergedTruth"
        #process.tauGenJets.GenParticles = "mergedTruth"
        #process.tauMatch.matched = "mergedTruth"


    #-- Taus ----------------------------------------------------------------------
    #some tau discriminators have been switched off during development. They can be switched on again...
    #setattr(process.patTaus.tauIDSources, "trackIsolation", cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolation"))
    #setattr(process.patTaus.tauIDSources, "ecalIsolation", cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolation"))
    #setattr(process.patTaus.tauIDSources, "byIsolation", cms.InputTag("shrinkingConePFTauDiscriminationByIsolation"))
    #setattr(process.patTaus.tauIDSources, "leadingPionPtCut", cms.InputTag("shrinkingConePFTauDiscriminationByLeadingPionPtCut"))
    #setattr(process.patTaus.tauIDSources, "trackIsolationUsingLeadingPion", cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion"))
    #setattr(process.patTaus.tauIDSources, "ecalIsolationUsingLeadingPion", cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion"))
    #setattr(process.patTaus.tauIDSources, "byIsolationUsingLeadingPion", cms.InputTag("shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion"))
    #setattr(process.patTaus.tauIDSources, "byTaNC", cms.InputTag("shrinkingConePFTauDiscriminationByTaNC"))
    #setattr(process.patTaus.tauIDSources, "byTaNCfrOnePercent", cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrOnePercent"))
    #setattr(process.patTaus.tauIDSources, "byTaNCfrHalfPercent", cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"))
    #setattr(process.patTaus.tauIDSources, "byTaNCfrQuarterPercent", cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"))
    #setattr(process.patTaus.tauIDSources, "byTaNCfrTenthPercent", cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrTenthPercent"))


    #-- Jet corrections -----------------------------------------------------------
    # L1 FastJet jet corrections
    # kt6PFJets for FastJet corrections are already run and placed before jetCorrection calculation
    # apply FastJet corrections only if demanded
    # TODO: Check if still necessary to switch here
    if ("L1FastJet" in jetMetCorrections):
        process.pfJets.doAreaFastjet = True
        process.pfJetsPF.doAreaFastjet = True

def loadPF2PAT(process,mcInfo,jetMetCorrections,extMatch,doSusyTopProjection,doType1MetCorrection,doType0MetCorrection,postfix):
    #-- PAT standard config -------------------------------------------------------
    process.load("PhysicsTools.PatAlgos.patSequences_cff")
    #-- Jet corrections -----------------------------------------------------------
    process.patJetCorrFactors.levels = jetMetCorrections 
    #-- PF2PAT config -------------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.pfTools import usePF2PAT
    usePF2PAT(process,runPF2PAT=True,
        jetAlgo             = 'AK5',
        runOnMC             = (mcInfo==1),
        postfix             = postfix,
        jetCorrections      = ('AK5PFchs', jetMetCorrections),
        typeIMetCorrections = doType1MetCorrection)

    if doType0MetCorrection:
        getattr(process,'patType1CorrectedPFMet'+postfix).srcType1Corrections = cms.VInputTag(
            cms.InputTag("patPFJetMETtype1p2Corr"+postfix,"type1"),
            cms.InputTag("patPFMETtype0Corr"+postfix)
        )


    #process.patJetsPF.embedGenJetMatch = False
    #process.patJetsPF.embedPFCandidates = False
    #drop tracks 
    process.patElectronsPF.embedTrack   = True
    process.patMuonsPF.embedTrack       = True
    process.electronMatchPF.maxDeltaR   = cms.double(0.2)
    process.electronMatchPF.maxDPtRel   = cms.double(999999.)
    process.electronMatchPF.checkCharge = False
    process.muonMatchPF.maxDeltaR       = cms.double(0.2)
    process.muonMatchPF.maxDPtRel       = cms.double(999999.)
    process.muonMatchPF.checkCharge     = False
    if extMatch:
        process.electronMatchPF.mcStatus        = cms.vint32(1,5)
        process.electronMatchPF.matched         = "mergedTruth"
        process.muonMatchPF.mcStatus            = cms.vint32(1,5)
        process.muonMatchPF.matched             = "mergedTruth"
        process.genParticlesForJets.src         = "mergedTruth"
        process.genParticlesForJetsNoMuNoNu.src = "mergedTruth"
        process.genParticlesForJetsNoNu.src     = "mergedTruth"
        process.patJetPartonMatchPF.matched     = "mergedTruth"
        process.patJetPartonsPF.src             = "mergedTruth"
        process.photonMatchPF.matched           = "mergedTruth"
        #process.tauGenJetsPF.GenParticles = "mergedTruth"
        #process.tauMatchPF.matched = "mergedTruth"
        
    #Remove jet pt cut
    #process.pfJetsPF.ptMin = 0.
    #include tau decay mode in pat::Taus (elese it will just be uninitialized)
    #process.patTausPF.addDecayMode = True
    #process.patTausPF.decayModeSrc = "shrinkingConePFTauDecayModeProducerPF" 

    #Set isolation cone to 0.3 for PF leptons
    # TODO: fix this for electrons and muons
    #process.pfElectrons.isolationValueMapsCharged = cms.VInputTag(cms.InputTag("elPFIsoValueCharged03PFId"))
    #process.pfElectrons.deltaBetaIsolationValueMap = cms.InputTag("elPFIsoValuePU03PFId")
    #process.pfElectrons.isolationValueMapsNeutral = cms.VInputTag(cms.InputTag("elPFIsoValueNeutral03PFId"), cms.InputTag("elPFIsoValueGamma03PFId"))
    #process.pfIsolatedElectrons.isolationValueMapsCharged = cms.VInputTag(cms.InputTag("elPFIsoValueCharged03PFId"))
    #process.pfIsolatedElectrons.deltaBetaIsolationValueMap = cms.InputTag("elPFIsoValuePU03PFId")
    #process.pfIsolatedElectrons.isolationValueMapsNeutral = cms.VInputTag(cms.InputTag("elPFIsoValueNeutral03PFId"), cms.InputTag("elPFIsoValueGamma03PFId"))
    
    #process.pfElectronsPF.isolationValueMapsCharged = cms.VInputTag(cms.InputTag("elPFIsoValueCharged03PFIdPF"))
    #process.pfElectronsPF.deltaBetaIsolationValueMap = cms.InputTag("elPFIsoValuePU03PFIdPF")
    #process.pfElectronsPF.isolationValueMapsNeutral = cms.VInputTag(cms.InputTag("elPFIsoValueNeutral03PFIdPF"), cms.InputTag("elPFIsoValueGamma03PFIdPF"))
    #process.pfIsolatedElectronsPF.isolationValueMapsCharged = cms.VInputTag(cms.InputTag("elPFIsoValueCharged03PFIdPF"))
    #process.pfIsolatedElectronsPF.deltaBetaIsolationValueMap = cms.InputTag("elPFIsoValuePU03PFIdPF")
    #process.pfIsolatedElectronsPF.isolationValueMapsNeutral = cms.VInputTag(cms.InputTag("elPFIsoValueNeutral03PFIdPF"), cms.InputTag("elPFIsoValueGamma03PFIdPF"))

    #process.pfMuons.isolationValueMapsCharged = cms.VInputTag(cms.InputTag("muPFIsoValueCharged03mu03"))
    #process.pfMuons.deltaBetaIsolationValueMap = cms.InputTag("muPFIsoValuePU03")
    #process.pfMuons.isolationValueMapsNeutral = cms.VInputTag(cms.InputTag("muPFIsoValueNeutral03"), cms.InputTag("muPFIsoValueGamma03"))
    #process.pfIsolatedMuons.isolationValueMapsCharged = cms.VInputTag(cms.InputTag("muPFIsoValueCharged03mu03"))
    #process.pfIsolatedMuons.deltaBetaIsolationValueMap = cms.InputTag("muPFIsoValuePU03")
    #process.pfIsolatedMuons.isolationValueMapsNeutral = cms.VInputTag(cms.InputTag("muPFIsoValueNeutral03"), cms.InputTag("muPFIsoValueGamma03"))

    #process.pfMuonsPF.isolationValueMapsCharged = cms.VInputTag(cms.InputTag("muPFIsoValueCharged03mu03PF"))
    #process.pfMuonsPF.deltaBetaIsolationValueMap = cms.InputTag("muPFIsoValuePU03PF")
    #process.pfMuonsPF.isolationValueMapsNeutral = cms.VInputTag(cms.InputTag("muPFIsoValueNeutral03PF"), cms.InputTag("muPFIsoValueGamma03PF"))
    #process.pfIsolatedMuonsPF.isolationValueMapsCharged = cms.VInputTag(cms.InputTag("muPFIsoValueCharged03mu03PF"))
    #process.pfIsolatedMuonsPF.deltaBetaIsolationValueMap = cms.InputTag("muPFIsoValuePU03PF")
    #process.pfIsolatedMuonsPF.isolationValueMapsNeutral = cms.VInputTag(cms.InputTag("muPFIsoValueNeutral03PF"), cms.InputTag("muPFIsoValueGamma03PF"))

    #-- Enable pileup sequence -------------------------------------------------------------
    #Vertices
    process.goodVertices = cms.EDFilter("VertexSelector",
        src = cms.InputTag("offlinePrimaryVertices"),
        cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
        filter = cms.bool(False),
    )

    process.pfPileUpPF.Vertices = "goodVertices"
    process.pfPileUpPF.Enable = True

    process.pfNoPileUpSequencePF.replace(process.pfPileUpPF,
                                         process.goodVertices + process.pfPileUpPF)

    if not doSusyTopProjection:
        return
    #-- Top projection selection -----------------------------------------------------------
    #Electrons
    #relax all selectors *before* pat-lepton creation
    process.pfElectronsFromVertexPF.dzCut = 9999.0
    process.pfElectronsFromVertexPF.d0Cut = 9999.0
    process.pfSelectedElectronsPF.cut = ""
    process.pfRelaxedElectronsPF = process.pfIsolatedElectronsPF.clone(isolationCut = 3.)
    process.pfIsolatedElectronsPF.isolationCut = 0.15
    
    process.pfElectronsFromGoodVertex = cms.EDFilter(
        "IPCutPFCandidateSelector",
        src = cms.InputTag("pfIsolatedElectronsPF"),  # PFCandidate source
        vertices = cms.InputTag("goodVertices"),  # vertices source
        d0Cut = cms.double(0.04),  # transverse IP
        dzCut = cms.double(1.),  # longitudinal IP
        d0SigCut = cms.double(99.),  # transverse IP significance
        dzSigCut = cms.double(99.),  # longitudinal IP significance
    )
   
    electronSelection =  "abs( eta ) < 2.5 & pt > 5"
    electronSelection += " & mva_e_pi > 0.4" # same as patElectron::mva()
    #electronSelection += " & (isEB & (sigmaIetaIeta < 0.024 & hadronicOverEm < 0.15) | isEE & (sigmaIetaIeta < 0.040 & hadronicOverEm < 0.10))" #caloIdVL
    #electronSelection += " & (isEB & (deltaPhiSuperClusterTrackAtVtx < 0.15 & deltaEtaSuperClusterTrackAtVtx < 0.01) | isEE & (deltaPhiSuperClusterTrackAtVtx < 0.10 & deltaEtaSuperClusterTrackAtVtx < 0.01))" #trkIdVL
    electronSelection += " & gsfTrackRef().isNonnull() & gsfTrackRef().trackerExpectedHitsInner().numberOfHits() <= 0"
    process.pfUnclusteredElectronsPF = cms.EDFilter( "GenericPFCandidateSelector",
        src = cms.InputTag("pfElectronsFromGoodVertex"), #pfSelectedElectronsPF
        cut = cms.string(electronSelection)
    )    
    process.pfElectronSequencePF.replace(process.pfIsolatedElectronsPF,
                                         process.pfIsolatedElectronsPF + 
                                         process.goodVertices * process.pfElectronsFromGoodVertex + 
                                         process.pfUnclusteredElectronsPF + process.pfRelaxedElectronsPF)
    process.patElectronsPF.pfElectronSource = "pfRelaxedElectronsPF"
    process.pfNoElectronPF.topCollection  = "pfUnclusteredElectronsPF"
    #Muons
    #relaxe built-in preselection
    process.pfMuonsFromVertexPF.dzCut = 9999.0
    process.pfMuonsFromVertexPF.d0Cut = 9999.0
    process.pfSelectedMuonsPF.cut = ""
    process.pfRelaxedMuonsPF = process.pfIsolatedMuonsPF.clone(isolationCut = 3)
    process.pfIsolatedMuonsPF.isolationCut = 0.15
    
    process.pfMuonsFromGoodVertex = cms.EDFilter(
        "IPCutPFCandidateSelector",
        src = cms.InputTag("pfIsolatedMuonsPF"),  # PFCandidate source
        vertices = cms.InputTag("goodVertices"),  # vertices source
        d0Cut = cms.double(0.02),  # transverse IP
        dzCut = cms.double(1.),  # longitudinal IP
        d0SigCut = cms.double(99.),  # transverse IP significance
        dzSigCut = cms.double(99.),  # longitudinal IP significance
    )
    muonSelection =  "abs( eta ) < 2.5 & pt > 5"
    #GlobalMuonPromptTight
    muonSelection += " & muonRef().isNonnull & muonRef().isGlobalMuon()"
    muonSelection += " & muonRef().isTrackerMuon() & muonRef().numberOfMatches > 1"
    muonSelection += " & muonRef().globalTrack().normalizedChi2() < 10"
    muonSelection += " & muonRef().track().numberOfValidHits() > 10"
    muonSelection += " & muonRef().globalTrack().hitPattern().numberOfValidMuonHits() > 0"
    muonSelection += " & muonRef().innerTrack().hitPattern().numberOfValidPixelHits() > 0"
    process.pfUnclusteredMuonsPF = cms.EDFilter( "GenericPFCandidateSelector",
        src = cms.InputTag("pfMuonsFromGoodVertex"), #pfSelectedMuonsPF
        cut = cms.string(muonSelection)
    )    
    process.pfMuonSequencePF.replace(process.pfIsolatedMuonsPF,
                                     process.pfIsolatedMuonsPF + 
                                     process.goodVertices * process.pfMuonsFromGoodVertex +
                                     process.pfUnclusteredMuonsPF + process.pfRelaxedMuonsPF)
    process.patMuonsPF.pfMuonSource  = "pfRelaxedMuonsPF"
    process.pfNoMuonPF.topCollection = "pfUnclusteredMuonsPF"
    #Taus
    # TODO: Fix taus in 52X
    #process.pfTausPF.discriminators = cms.VPSet()
    #process.pfUnclusteredTausPF = process.pfTausPF.clone(
    #    cut = cms.string("pt < 0")
    #)
    #process.pfTauSequencePF.replace(process.pfTausPF, process.pfTausPF+ process.pfUnclusteredTausPF)
    #process.pfNoTauPF.topCollection = "pfUnclusteredTausPF"
    

def loadPATTriggers(process,HLTMenu,theJetNames,electronMatches,muonMatches,tauMatches,jetMatches,photonMatches):
    #-- Trigger matching ----------------------------------------------------------
    def pfSwitchOnTriggerMatchEmbedding(process, matches, src, embedder, sequence='patDefaultSequencePF'):
        setattr(process,src.replace('PF','TriggerMatchPF'),getattr(process,embedder).clone(src=src, matches=matches))
        theSequence = getattr(process,sequence)
        theSequence += getattr(process,src.replace('PF','TriggerMatchPF'))
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger, switchOnTriggerMatchEmbedding
    switchOnTrigger(process, triggerProducer='patTrigger', triggerEventProducer='patTriggerEvent', sequence='patDefaultSequence', hltProcess=HLTMenu)
    process.patTriggerPF = process.patTrigger.clone()
    process.patTriggerEventPF = process.patTriggerEvent.clone()
    process.patDefaultSequencePF += process.patTriggerPF 
    process.patDefaultSequencePF += process.patTriggerEventPF 
    switchOnTrigger(process, triggerProducer='patTriggerPF', triggerEventProducer='patTriggerEventPF', sequence='patDefaultSequencePF', hltProcess=HLTMenu)
    #Electrons
    from PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi import cleanElectronTriggerMatchHLTEle20SWL1R
    process.patElectronMatch = cleanElectronTriggerMatchHLTEle20SWL1R.clone(matchedCuts = cms.string( electronMatches ))
    process.patElectronMatchPF = cleanElectronTriggerMatchHLTEle20SWL1R.clone(matchedCuts = cms.string( electronMatches ), src='selectedPatElectronsPF')
    process.patDefaultSequencePF += process.patElectronMatchPF
    switchOnTriggerMatchEmbedding( process, ['patElectronMatch'], hltProcess=HLTMenu)
    pfSwitchOnTriggerMatchEmbedding( process, ['patElectronMatchPF'], 'selectedPatElectronsPF', 'cleanPatElectronsTriggerMatch' )
    #Muons
    from PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi import cleanMuonTriggerMatchHLTMu9
    process.patMuonMatch = cleanMuonTriggerMatchHLTMu9.clone(matchedCuts = cms.string( muonMatches ))
    process.patMuonMatchPF = cleanMuonTriggerMatchHLTMu9.clone(matchedCuts = cms.string( muonMatches ),src = 'selectedPatMuonsPF',matched='patTriggerPF')
    process.patDefaultSequencePF += process.patMuonMatchPF
    switchOnTriggerMatchEmbedding( process, ['patMuonMatch'], hltProcess=HLTMenu)
    pfSwitchOnTriggerMatchEmbedding( process, ['patMuonMatchPF'], 'selectedPatMuonsPF', 'cleanPatMuonsTriggerMatch' )
    #Photons
    from PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi import cleanPhotonTriggerMatchHLTPhoton20CleanedL1R
    process.patPhotonMatch = cleanPhotonTriggerMatchHLTPhoton20CleanedL1R.clone(matchedCuts = cms.string( photonMatches ))
    switchOnTriggerMatchEmbedding( process, ['patPhotonMatch'], hltProcess=HLTMenu)
    #Jets
    from PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi import cleanJetTriggerMatchHLTJet15U
    process.patJetMatchAK5Calo = cleanJetTriggerMatchHLTJet15U.clone(matchedCuts = cms.string( jetMatches ),src='cleanPatJetsAK5Calo')
    switchOnTriggerMatchEmbedding( process, ['patJetMatchAK5Calo'], hltProcess=HLTMenu)
    for jetType in theJetNames:
        setattr(process,'patJetMatch'+jetType,cleanJetTriggerMatchHLTJet15U.clone(matchedCuts = cms.string( jetMatches ),src = 'cleanPatJets'+jetType))
    process.patJetMatchPF = cleanJetTriggerMatchHLTJet15U.clone(src='selectedPatJetsPF', matchedCuts = cms.string( jetMatches ))
    process.patDefaultSequencePF += process.patJetMatchPF
    for jetType in theJetNames:
        switchOnTriggerMatchEmbedding( process, ['patJetMatch'+jetType], hltProcess=HLTMenu)
    pfSwitchOnTriggerMatchEmbedding( process, ['patJetMatchPF'], 'selectedPatJetsPF', 'cleanPatJetsAK5CaloTriggerMatch' )
    #Taus
    from PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi import cleanTauTriggerMatchHLTDoubleLooseIsoTau15
    process.patTauMatch = cleanTauTriggerMatchHLTDoubleLooseIsoTau15.clone(matchedCuts = cms.string( tauMatches ))
    process.patTauMatchPF = cleanTauTriggerMatchHLTDoubleLooseIsoTau15.clone(src='selectedPatTausPF', matchedCuts = cms.string( tauMatches ))
    process.patDefaultSequencePF += process.patTauMatchPF
    switchOnTriggerMatchEmbedding( process, ['patTauMatch'], hltProcess=HLTMenu)
    pfSwitchOnTriggerMatchEmbedding( process, ['patTauMatchPF'], 'selectedPatTausPF', 'cleanPatTausTriggerMatch' )

def loadType1METSequence(process):
    process.load("JetMETCorrections.Type1MET.pfMETCorrections_cff")
    ## Type 0?
    # process.pfType1CorrectedMet.srcCHSSums = cms.VInputTag(cms.InputTag("pfchsMETcorr","type0"))
    # process.pfType1CorrectedMet.applyType2Corrections = cms.bool(False)
    # process.pfType1CorrectedMet.type0Rsoft = cms.double(0.6)
    # process.pfType1CorrectedMet.applyType0Corrections = cms.bool(True)
    process.Type1METSequence = cms.Sequence(process.producePFMETCorrections)


def addTypeIIMet(process) :
    # Add reco::MET with Type II correction 
    from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import metJESCorAK5CaloJet
    process.metJESCorAK5CaloJetTypeII = metJESCorAK5CaloJet.clone()
    process.metJESCorAK5CaloJetTypeII.useTypeII = True
    process.metJESCorAK5CaloJetTypeII.hasMuonsCorr = False
    # Add muon corrections for above II reco::MET
    process.metJESCorAK5CaloJetMuonsTypeII = process.metJESCorAK5CaloJetMuons.clone(
        uncorMETInputTag = cms.InputTag('metJESCorAK5CaloJetTypeII')
        )
    # Add to recoLayer0 sequence
    process.patMETCorrections.replace(
        process.metJESCorAK5CaloJet,
        (process.metJESCorAK5CaloJetTypeII*
         process.metJESCorAK5CaloJetMuonsTypeII)+
        process.metJESCorAK5CaloJet
        )
    # Add pat::MET with Type II correction
    process.patMETsAK5CaloTypeII = process.patMETs.clone(
        metSource = cms.InputTag("metJESCorAK5CaloJetMuonsTypeII")
        )
    # Add to producersLayer1 sequence
    process.patDefaultSequence.replace(
        process.patMETsAK5Calo,
        process.patMETsAK5Calo+
        process.patMETsAK5CaloTypeII
        )

def addTagInfos(process,jetMetCorrections):
    from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection
    switchJetCollection( process,
                     jetCollection=cms.InputTag('ak5CaloJets'),
                     jetCorrLabel=('AK5Calo', jetMetCorrections))

def addSUSYJetCollection(process,jetMetCorrections,jets = 'IC5Calo',mcVersion='',doJTA=True,doType1MET=True,doJetID=True,jetIdLabel=None):
    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    algorithm = jets[0:3]
    type = jets[3:len(jets)]
    jetCorrLabel = (jets,cms.vstring(jetMetCorrections))
    if 'IC' in algorithm: collection = algorithm.replace('IC','iterativeCone')
    elif 'SC' in algorithm: collection = algorithm.replace('SC','sisCone')
    elif 'AK' in algorithm: collection = algorithm.replace('AK','ak')
    elif 'KT' in algorithm: collection = algorithm.replace('KT','kt')
    else: raise ValueError, "Unknown jet algorithm: %s" % (jets)
    jetIdLabel = algorithm.lower()
    if type == 'Calo':
        jetCollection = '%(collection)sCaloJets' % locals()
        if not 'AK7' in algorithm:
            doType1MET = True
    elif type == 'PF':
        jetCollection = '%(collection)sPFJets' % locals()
        doJetID = False
    elif type == 'JPT':
        if 'IC' in algorithm: collectionJPT = algorithm.replace('IC','Icone')
        elif 'SC' in algorithm: collectionJPT = algorithm.replace('SC','Siscone')
        elif 'AK' in algorithm: collectionJPT = algorithm.replace('AK','AntiKt')
        else: raise ValueError, "Unknown jet algorithm: %s" % (jets)
        jetCollection = 'JetPlusTrackZSPCorJet%(collectionJPT)s' % locals()
    elif type == 'Track':
        jetCollection = '%(collection)sTrackJets' % locals()
        jetCorrLabel = None
        doJetID = False
    else: raise ValueError, "Unknown jet type: %s" % (jets)
    
    addJetCollection(process, cms.InputTag(jetCollection),
                     algorithm, type,
                     doJTA            = doJTA,
                     doBTagging       = True,
                     jetCorrLabel     = jetCorrLabel,
                     doType1MET       = doType1MET,
                     doL1Cleaning     = True,
                     doL1Counters     = True,
                     doJetID          = doJetID,
                     jetIdLabel       = jetIdLabel,
                     genJetCollection = cms.InputTag('%(collection)sGenJets' % locals())
                     )

def addJetMET(process,theJetNames,jetMetCorrections,mcVersion):
    #-- Extra Jet/MET collections -------------------------------------------------
    # Add a few jet collections...
    for jetName in theJetNames:
        addSUSYJetCollection(process,jetMetCorrections,jetName,mcVersion)
    
    #-- Tune contents of jet collections  -----------------------------------------
    theJetNames.append('')
    for jetName in theJetNames:
        module = getattr(process,'patJets'+jetName)
        module.addTagInfos = False    # Remove tag infos
        module.embedGenJetMatch = False # Only keep reference, since we anyway keep the genJet collections
        #module.embedCaloTowers = True # To drop calo towers
    theJetNames.pop()
    
    # Add tcMET
    from PhysicsTools.PatAlgos.tools.metTools import addTcMET #, addPfMET
    addTcMET(process,'TC')
    #addPfMET(process,'PF') #is in PF2PAT

    # Rename default jet collection for uniformity
    process.cleanPatJetsAK5Calo = process.cleanPatJets
    process.patMETsAK5Calo      = process.patMETs
    
    # TODO: fix type2 MET in 52X
    #addTypeIIMet(process)

    # Modify subsequent modules
    process.patHemispheres.patJets = process.cleanPatJetsAK5Calo.label()
    process.countPatJets.src       = process.cleanPatJetsAK5Calo.label()
    
    # Modify counters' input
    process.patCandidateSummary.candidates.remove(cms.InputTag('patMETs'))
    process.patCandidateSummary.candidates.append(cms.InputTag('patMETsAK5Calo'))
    process.patCandidateSummary.candidates.append(cms.InputTag('patMHTsAK5Calo'))
    process.cleanPatCandidateSummary.candidates.remove(cms.InputTag('cleanPatJets'))
    process.cleanPatCandidateSummary.candidates.append(cms.InputTag('cleanPatJetsAK5Calo'))
    # Add new jet collections to counters (MET done automatically)
    for jets in theJetNames: 
        process.patCandidateSummary.candidates.append(cms.InputTag('patJets'+jets))
        process.selectedPatCandidateSummary.candidates.append(cms.InputTag('selectedPatJets'+jets))
        process.cleanPatCandidateSummary.candidates.append(cms.InputTag('cleanPatJets'+jets))
    
def removeMCDependence( process ):
    #-- Remove MC dependence ------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.coreTools import removeMCMatching
    removeMCMatching(process, ['All'])

def loadSusyValidation(process):
    process.load("JetMETCorrections.Configuration.JetCorrectionProducers_cff")
    process.load("DQM.Physics.susyValidation_cfi")
    process.load("DQMServices.Components.MEtoEDMConverter_cfi")
    process.load("DQMServices.Core.DQM_cfg")
    process.load("DQMServices.Components.DQMEnvironment_cfi")
    process.DQMStore = cms.Service("DQMStore")
    process.DQMStore.collateHistograms = cms.untracked.bool(True)
    process.options = cms.untracked.PSet(
        fileMode = cms.untracked.string('NOMERGE')
    )

def getSUSY_pattuple_outputCommands( process ):
    from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent, patExtraAodEventContent, patTriggerEventContent, patTriggerStandAloneEventContent, patEventContentTriggerMatch
    keepList = []
    susyAddEventContent = [ # PAT Objects
    #'keep *_triggerMatched*_*_*',         
    # Keep PF2PAT output
    'keep *_selectedPatMuonsPF_*_*',         
    'keep *_selectedPatElectronsPF_*_*',         
    'keep *_selectedPatTausPF_*_*',         
    'keep *_selectedPatJetsPF_*_*',
    #L1 trigger info         
    'keep L1GlobalTriggerObjectMapRecord_*_*_*',
    'keep L1GlobalTriggerReadoutRecord_*_*_*',
    # Generator information
    'keep recoGenJets_*GenJets*_*_*',
    'keep recoGenMETs_*_*_*',
    #Number of processed events
    'keep edmMergeableCounter_eventCountProducer_*_*',
    'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*',
    #'keep recoTrackJets_ak5TrackJets_*_*',
    'keep *_electronMergedSeeds_*_*',
    'keep *_Conversions_*_*',
    'keep recoPFCandidates_particleFlow_*_*',
    #'keep recoSuperClusters_corrected*_*_*',
    #'keep recoSuperClusters_pfElectronTranslator_*_*',
    #'keep *_gsfElectronCores_*_*',    #Keep electron core
    #'keep *_photonCore_*_*',        #Keep electron core
    'keep recoConversions_conversions_*_*',
    'keep recoTracks_*onversions_*_*',
    'keep HcalNoiseSummary_*_*_*', #Keep the one in RECO
    'keep *BeamHaloSummary_*_*_*',
    # Keep Gap Vertices for comparison
    'keep *_offlinePrimaryVerticesGap_*_*',
    'keep *_offlinePrimaryVerticesGapWithBS_*_*',
    #DQM
    'keep *_MEtoEDMConverter_*_PAT',
    'drop recoTracks_generalTracks*_*_*',
    'drop *_towerMaker_*_*',
    'keep *_pfType1CorrectedMet*_*_*',
    ]
    keepList.extend(patEventContent)
    keepList.extend(patExtraAodEventContent)
    keepList.extend(patTriggerEventContent)
    keepList.extend(patEventContentTriggerMatch)
    keepList.extend(susyAddEventContent)
    return keepList


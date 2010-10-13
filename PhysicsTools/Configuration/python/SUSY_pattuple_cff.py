#
#  SUSY-PAT configuration fragment
#
#  PAT configuration for the SUSY group - 38X series
#  More information here:
#  https://twiki.cern.ch/twiki/bin/view/CMS/SusyPatLayer1DefV9


import FWCore.ParameterSet.Config as cms

def addDefaultSUSYPAT(process,mcInfo=True,HLTMenu='HLT',JetMetCorrections='Spring10',mcVersion='',theJetNames = ['IC5Calo','AK5JPT'],doValidation=False,extMatch=False,doSusyTopProjection=False,electronMatches=[],muonMatches=[],tauMatches=[],jetMatches=[],photonMatches=[]):
    loadPF2PAT(process,mcInfo,JetMetCorrections,extMatch,doSusyTopProjection,'PF')
    if not mcInfo:
	removeMCDependence(process)
    loadMCVersion(process,mcVersion,mcInfo)
    loadPAT(process,JetMetCorrections,extMatch)
    addJetMET(process,theJetNames,mcVersion)
    loadPATTriggers(process,HLTMenu,theJetNames,electronMatches,muonMatches,tauMatches,jetMatches,photonMatches)

    #-- Counter for the number of processed events --------------------------------
    process.eventCountProducer = cms.EDProducer("EventCountProducer")

    # Full path
    process.susyPatDefaultSequence = cms.Sequence( process.eventCountProducer 
                                                   * process.patDefaultSequence * process.patPF2PATSequencePF
                                                   )

    if mcInfo and extMatch:
        extensiveMatching(process)
        process.susyPatDefaultSequence.replace(process.patDefaultSequence, process.extensiveMatching+process.patDefaultSequence)
    
    if mcVersion == '35x' and 'JPT' in ''.join(theJetNames): 
    	process.susyPatDefaultSequence.replace(process.eventCountProducer, process.eventCountProducer * process.recoJPTJets)
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


def loadMCVersion(process, mcVersion, mcInfo):
    #-- To be able to run on 35X input samples ---------------------------------------
    from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run36xOn35xInput
    if not mcVersion:
	return
    elif mcVersion == '35x': 
	run36xOn35xInput(process)
	if mcInfo:
		run36xOnReRecoMC(process)
    	#-- Jet plus tracks are in RECO in 36X, but not in 35X-----------------------
	process.load("RecoJets.Configuration.RecoJPTJets_cff")
    else: raise ValueError, "Unknown MC version: %s" % (mcVersion)


def loadPAT(process,JetMetCorrections,extMatch):
    #-- Changes for electron and photon ID ----------------------------------------
    # Turn off photon-electron cleaning (i.e., flag only)
    process.cleanPatPhotons.checkOverlaps.electrons.requireNoOverlaps = False

    # Remove embedding of superClusters, will keep entire superCluster collection
    process.patElectrons.embedSuperCluster = False
    process.patPhotons.embedSuperCluster   = False

    #include tau decay mode in pat::Taus (elese it will just be uninitialized)
    #decay modes are dropped and have to be redone, this is a bit dangorous since the decay modes insered are *not* the ones used in RECO
    process.patTaus.addDecayMode = True
    process.makePatTaus.replace( process.patTaus, process.shrinkingConePFTauDecayModeProducer + process.patTaus )

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
    process.makePatElectrons.replace(process.patElectrons,process.simpleEleIdSequence+process.patElectrons)
    
    #-- Tuning of Monte Carlo matching --------------------------------------------
    # Also match with leptons of opposite charge
    process.electronMatch.checkCharge = False
    process.electronMatch.maxDeltaR   = cms.double(0.2)
    process.electronMatch.maxDPtRel   = cms.double(999999.)
    process.muonMatch.checkCharge     = False
    process.muonMatch.maxDeltaR       = cms.double(0.2)
    process.muonMatch.maxDPtRel       = cms.double(999999.)
    process.tauMatch.checkCharge      = False
    process.tauMatch.maxDeltaR        = cms.double(0.3)
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
        process.tauGenJets.GenParticles = "mergedTruth"
        process.tauMatch.matched = "mergedTruth"

    #-- Jet corrections -----------------------------------------------------------
    process.patJetCorrFactors.corrSample = JetMetCorrections 

def loadPF2PAT(process,mcInfo,JetMetCorrections,extMatch,doSusyTopProjection,postfix):
    #-- PAT standard config -------------------------------------------------------
    process.load("PhysicsTools.PatAlgos.patSequences_cff")
    #-- Jet corrections -----------------------------------------------------------
    process.patJetCorrFactors.corrSample = JetMetCorrections 
    #-- PF2PAT config -------------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.pfTools import usePF2PAT
    usePF2PAT(process,runPF2PAT=True, jetAlgo='AK5',runOnMC=mcInfo,postfix=postfix)
    process.patJetsPF.embedGenJetMatch = False
    process.patJetsPF.embedPFCandidates = False
    process.electronMatchPF.maxDeltaR   = cms.double(0.2)
    process.electronMatchPF.maxDPtRel   = cms.double(999999.)
    process.electronMatchPF.checkCharge = False
    process.muonMatchPF.maxDeltaR   = cms.double(0.2)
    process.muonMatchPF.maxDPtRel   = cms.double(999999.)
    process.muonMatchPF.checkCharge = False
    if extMatch:
        process.electronMatchPF.mcStatus = cms.vint32(1,5)
        process.electronMatchPF.matched = "mergedTruth"
        process.muonMatchPF.mcStatus = cms.vint32(1,5)
        process.muonMatchPF.matched = "mergedTruth"
        process.genParticlesForJets.src = "mergedTruth"
        process.genParticlesForJetsNoMuNoNu.src = "mergedTruth"
        process.genParticlesForJetsNoNu.src = "mergedTruth"
        process.patJetPartonMatchPF.matched = "mergedTruth"
        process.patJetPartonsPF.src = "mergedTruth"
        process.photonMatchPF.matched = "mergedTruth"
        process.tauGenJetsPF.GenParticles = "mergedTruth"
        process.tauMatchPF.matched = "mergedTruth"
        
    #Remove jet pt cut
    process.pfJetsPF.ptMin = 0.
    #include tau decay mode in pat::Taus (elese it will just be uninitialized)
    process.patTausPF.addDecayMode = True
    process.patTausPF.decayModeSrc = "shrinkingConePFTauDecayModeProducerPF"    

    if not doSusyTopProjection:
        return
    #-- Top projection selection -----------------------------------------------------------
    #Vertices
    process.goodVertices = cms.EDFilter("VertexSelector",
        src = cms.InputTag("offlinePrimaryVertices"),
        cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
        filter = cms.bool(False),
    )
    #Electrons
    process.pfRelaxedElectronsPF = process.pfIsolatedElectronsPF.clone(combinedIsolationCut = 3.)
    process.pfIsolatedElectronsPF.combinedIsolationCut = 0.3
    process.pfElectronsFromVertex = cms.EDFilter(
        "IPCutPFCandidateSelector",
        src = cms.InputTag("pfIsolatedElectronsPF"),  # PFCandidate source
        vertices = cms.InputTag("goodVertices"),  # vertices source
        d0Cut = cms.double(0.04),  # transverse IP
        dzCut = cms.double(1.),  # longitudinal IP
        d0SigCut = cms.double(99.),  # transverse IP significance
        dzSigCut = cms.double(99.),  # longitudinal IP significance
    )
    electronSelection =  "abs( eta ) < 2.5 & pt > 5"
    electronSelection += " & mva_e_pi > 0.6" # same as patElectron::mva()
    electronSelection += " & gsfTrackRef().isNonnull() & gsfTrackRef().trackerExpectedHitsInner().numberOfHits() > 1"
    process.pfUnclusteredElectronsPF = cms.EDFilter( "GenericPFCandidateSelector",
        src = cms.InputTag("pfIsolatedElectronsPF"), #pfSelectedElectronsPF
        cut = cms.string(electronSelection)
    )    
    process.pfElectronSequencePF.replace(process.pfIsolatedElectronsPF,
                                         process.pfIsolatedElectronsPF + 
                                         process.goodVertices * process.pfElectronsFromVertex + 
                                         process.pfUnclusteredElectronsPF + process.pfRelaxedElectronsPF)
    process.patElectronsPF.pfElectronSource = "pfRelaxedElectronsPF"
    process.pfNoElectronPF.topCollection  = "pfUnclusteredElectronsPF"
    #Muons
    process.pfRelaxedMuonsPF = process.pfIsolatedMuonsPF.clone(combinedIsolationCut = 3)
    process.pfIsolatedMuonsPF.combinedIsolationCut = 0.25
    process.pfMuonsFromVertex = cms.EDFilter(
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
        src = cms.InputTag("pfIsolatedMuonsPF"), #pfSelectedMuonsPF
        cut = cms.string(muonSelection)
    )    
    process.pfMuonSequencePF.replace(process.pfIsolatedMuonsPF,
                                     process.pfIsolatedMuonsPF + 
                                     process.goodVertices * process.pfMuonsFromVertex +
                                     process.pfUnclusteredMuonsPF + process.pfRelaxedMuonsPF)
    process.patMuonsPF.pfMuonSource  = "pfRelaxedMuonsPF"
    process.pfNoMuonPF.topCollection = "pfUnclusteredMuonsPF"
    #Taus
    process.pfTausPF.discriminators = cms.VPSet()
    process.pfUnclusteredTausPF = process.pfTausPF.clone(
        cut = cms.string("pt < 0")
    )
    process.pfTauSequencePF.replace(process.pfTausPF, process.pfTausPF+ process.pfUnclusteredTausPF)
    process.pfNoTauPF.topCollection = "pfUnclusteredTausPF"
    

def loadPATTriggers(process,HLTMenu,theJetNames,electronMatches,muonMatches,tauMatches,jetMatches,photonMatches):
    #-- Trigger matching ----------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
    switchOnTrigger( process )
    # If we have to rename the default trigger menu
    process.patTrigger.processName = HLTMenu
    process.patTriggerEvent.processName = HLTMenu
    #Add dummy sequence
    process.patTriggerMatcher = cms.Sequence(process.patJets)
    process.patTriggerMatcherPF = cms.Sequence(process.patJetsPF)
    process.patTriggerSequencePF = cms.Sequence(process.patTrigger*process.patTriggerMatcherPF*process.patTriggerEvent)
    #Electrons
    for i in range(len(electronMatches)):
        setattr(process,'electronTriggerMatch'+electronMatches[i].replace('_',''),process.electronTriggerMatchHLTEle15LWL1R.clone(pathNames=[electronMatches[i]]))
        process.patTriggerMatcher += getattr(process,'electronTriggerMatch'+electronMatches[i].replace('_',''))
        setattr(process,'electronTriggerMatch'+electronMatches[i].replace('_','')+'PF',process.electronTriggerMatchHLTEle15LWL1R.clone(pathNames=[electronMatches[i]],src = 'selectedPatElectronsPF'))
        process.patTriggerMatcherPF += getattr(process,'electronTriggerMatch'+electronMatches[i].replace('_','')+'PF')
        electronMatches[i] = 'electronTriggerMatch'+electronMatches[i].replace('_','')
    process.triggerMatchedPatElectrons = process.cleanPatElectronsTriggerMatch.clone(matches=electronMatches)
    process.triggerMatchedPatElectronsPF = process.triggerMatchedPatElectrons.clone(src='selectedPatElectronsPF',matches=[i+'PF' for i in electronMatches])
    process.patTriggerSequence += process.triggerMatchedPatElectrons
    process.patTriggerSequencePF += process.triggerMatchedPatElectronsPF
    #Muons
    for i in range(len(muonMatches)):
        setattr(process,'muonTriggerMatch'+muonMatches[i].replace('_',''),process.muonTriggerMatchHLTMu3.clone(pathNames=[muonMatches[i]]))
        process.patTriggerMatcher += getattr(process,'muonTriggerMatch'+muonMatches[i].replace('_',''))
        setattr(process,'muonTriggerMatch'+muonMatches[i].replace('_','')+'PF',process.muonTriggerMatchHLTMu3.clone(pathNames=[muonMatches[i]],src = 'selectedPatMuonsPF'))
        process.patTriggerMatcherPF += getattr(process,'muonTriggerMatch'+muonMatches[i].replace('_','')+'PF')
        muonMatches[i] = 'muonTriggerMatch'+muonMatches[i].replace('_','')
    process.triggerMatchedPatMuons = process.cleanPatMuonsTriggerMatch.clone(matches = muonMatches)
    process.triggerMatchedPatMuonsPF = process.triggerMatchedPatMuons.clone(src='selectedPatMuonsPF',matches=[i+'PF' for i in muonMatches])
    process.patTriggerSequence += process.triggerMatchedPatMuons
    process.patTriggerSequencePF += process.triggerMatchedPatMuonsPF
    #Photons
    for i in range(len(photonMatches)):
        setattr(process,'photonTriggerMatch'+photonMatches[i].replace('_',''),process.electronTriggerMatchHLTEle15LWL1R.clone(pathNames=[photonMatches[i]],src = 'cleanPatPhotons',filterIdsEnum = cms.vstring('TriggerPhoton')))
        process.patTriggerMatcher += getattr(process,'photonTriggerMatch'+photonMatches[i].replace('_',''))
        photonMatches[i] = 'photonTriggerMatch'+photonMatches[i].replace('_','')
    process.triggerMatchedPatPhotons = process.cleanPatPhotonsTriggerMatch.clone(matches = photonMatches)
    process.patTriggerSequence += process.triggerMatchedPatPhotons
    #Jets
    for i in range(len(jetMatches)):
        setattr(process,'jetTriggerMatch'+jetMatches[i].replace('_','')+'AK5Calo',process.muonTriggerMatchHLTMu3.clone(pathNames=[jetMatches[i]],src = 'cleanPatJetsAK5Calo',filterIdsEnum = cms.vstring('TriggerJet')))
        process.patTriggerMatcher += getattr(process,'jetTriggerMatch'+jetMatches[i].replace('_','')+'AK5Calo')
        for jetType in theJetNames:
            setattr(process,'jetTriggerMatch'+jetMatches[i].replace('_','')+jetType,process.muonTriggerMatchHLTMu3.clone(pathNames=[jetMatches[i]],src = 'cleanPatJets'+jetType,filterIdsEnum = cms.vstring('TriggerJet')))
            process.patTriggerMatcher += getattr(process,'jetTriggerMatch'+jetMatches[i].replace('_','')+jetType)
        setattr(process,'jetTriggerMatch'+jetMatches[i].replace('_','')+'PF',process.muonTriggerMatchHLTMu3.clone(pathNames=[jetMatches[i]],src = 'selectedPatJetsPF',filterIdsEnum = cms.vstring('TriggerJet')))
        process.patTriggerMatcherPF += getattr(process,'jetTriggerMatch'+jetMatches[i].replace('_','')+'PF')
        jetMatches[i] = 'jetTriggerMatch'+jetMatches[i].replace('_','')+'AK5Calo'
    process.triggerMatchedPatJetsAK5Calo = process.cleanPatJetsTriggerMatch.clone(matches = jetMatches,src = 'cleanPatJetsAK5Calo')
    process.patTriggerSequence += process.triggerMatchedPatJetsAK5Calo
    for jetType in theJetNames:
        setattr(process,'triggerMatchedPatJets'+jetType,process.triggerMatchedPatJetsAK5Calo.clone(src='cleanPatJets'+jetType, matches=[i.replace('AK5Calo',jetType) for i in jetMatches]))
        process.patTriggerSequence += getattr(process,'triggerMatchedPatJets'+jetType)
    process.triggerMatchedPatJetsPF = process.triggerMatchedPatJetsAK5Calo.clone(src='selectedPatJetsPF', matches=[i.replace('AK5Calo','PF') for i in jetMatches])
    process.patTriggerSequencePF += process.triggerMatchedPatJetsPF
    #Taus
    for i in range(len(tauMatches)):
        setattr(process,'tauTriggerMatch'+tauMatches[i].replace('_',''),process.muonTriggerMatchHLTMu3.clone(pathNames=[tauMatches[i]],src = 'cleanPatTaus',filterIdsEnum = cms.vstring('TriggerTau')))
        process.patTriggerMatcher += getattr(process,'tauTriggerMatch'+tauMatches[i].replace('_',''))
        setattr(process,'tauTriggerMatch'+tauMatches[i].replace('_','')+'PF',process.muonTriggerMatchHLTMu3.clone(pathNames=[tauMatches[i]],src = 'selectedPatTausPF',filterIdsEnum = cms.vstring('TriggerTau')))
        process.patTriggerMatcherPF += getattr(process,'tauTriggerMatch'+tauMatches[i].replace('_','')+'PF')
        tauMatches[i] = 'tauTriggerMatch'+tauMatches[i].replace('_','')
    process.triggerMatchedPatTaus = process.cleanPatTausTriggerMatch.clone(matches = tauMatches)
    process.triggerMatchedPatTausPF = process.triggerMatchedPatTaus.clone(src='selectedPatTausPF',matches=[i+'PF' for i in tauMatches])
    process.patTriggerSequence += process.triggerMatchedPatTaus
    process.patTriggerSequencePF += process.triggerMatchedPatTausPF
    process.patTriggerEvent.patTriggerMatches = electronMatches+muonMatches+tauMatches+jetMatches
    #Remove dummy from sequence
    process.patTriggerMatcher.remove(process.patJets)
    process.patTriggerMatcherPF.remove(process.patJetsPF)
    #Add PF trigger matching
    process.patDefaultSequencePF += process.patTriggerSequencePF

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
    process.makePatMETs.replace(
        process.patMETs,
        process.patMETs+
        process.patMETsAK5CaloTypeII
        )

def addSUSYJetCollection(process,jets = 'IC5Calo',mcVersion='',doJTA=False,doType1MET=False,doJetID=True,jetIdLabel=None):
    if mcVersion == '35x':
        from PhysicsTools.PatAlgos.tools.cmsswVersionTools import addJetCollection35X as addJetCollection
    else:
	from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    algorithm = jets[0:3]
    type = jets[3:len(jets)]
    jetCorrLabel = (algorithm,type)
    if 'IC' in algorithm: collection = algorithm.replace('IC','iterativeCone')
    elif 'SC' in algorithm: collection = algorithm.replace('SC','sisCone')
    elif 'AK' in algorithm: collection = algorithm.replace('AK','ak')
    elif 'KT' in algorithm: collection = algorithm.replace('KT','kt')
    else: raise ValueError, "Unknown jet algorithm: %s" % (jets)
    jetIdLabel = algorithm.lower()
    if type == 'Calo':
	jetCollection = '%(collection)sCaloJets' % locals()
        doJTA = True
	if not 'AK7' in algorithm:
		doType1MET = True
    elif type == 'PF':
	jetCollection = '%(collection)sPFJets' % locals()
        doJTA = True
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

def addJetMET(process,theJetNames,mcVersion):
    #-- Extra Jet/MET collections -------------------------------------------------
    # Add a few jet collections...
    for jetName in theJetNames:
    	addSUSYJetCollection(process,jetName,mcVersion)
    
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
    #process.patMHTsAK5Calo      = process.patMHTs
    addTypeIIMet(process)

    # Modify subsequent modules
    process.patHemispheres.patJets = process.cleanPatJetsAK5Calo.label()
    process.countPatJets.src       = process.cleanPatJetsAK5Calo.label()
    
    # Add MHT (inserted until officially suported)
    #from PhysicsTools.PatAlgos.producersLayer1.mhtProducer_cff import makePatMHTs, patMHTs
    #process.countPatCandidates.replace(process.countPatJets, process.countPatJets + process.makePatMHTs)
    #process.patMHTs.jetTag      = 'patJets'
    #process.patMHTs.electronTag = 'patElectrons'
    #process.patMHTs.muonTag     = 'patMuons'
    #process.patMHTs.tauTag      = 'patTaus'
    #process.patMHTs.photonTag   = 'patPhotons'

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
	'keep recoTrackJets_ak5TrackJets_*_*',
	'keep *_electronMergedSeeds_*_*',
	'keep *_Conversions_*_*',
	'keep recoPFCandidates_particleFlow_*_*',
        'keep recoSuperClusters_corrected*_*_*',
	'keep recoSuperClusters_pfElectronTranslator_*_*',
        'keep *_gsfElectronCores_*_*',    #Keep electron core
        'keep *_photonCore_*_*',        #Keep electron core
        'keep recoConversions_conversions_*_*',
        'keep recoTracks_*onversions_*_*',
        'keep HcalNoiseSummary_*_*_*', #Keep the one in RECO
	'keep *BeamHaloSummary_*_*_*',
	#DQM
	'keep *_MEtoEDMConverter_*_PAT',
	#'drop *_towerMaker_*_*'
        ] 
	keepList.extend(patEventContent)
	keepList.extend(patExtraAodEventContent)
	keepList.extend(patTriggerEventContent)
	keepList.extend(patEventContentTriggerMatch)
	keepList.extend(susyAddEventContent)
	return keepList


def run36xOnReRecoMC( process ):
    """
    ------------------------------------------------------------------
    running GenJets for ak5 and ak7

    process : process
    genJets : which gen jets to run
    ------------------------------------------------------------------    
    """
    print "*********************************************************************"
    print "NOTE TO USER: when running on 31X samples re-recoed in 3.5.6         "
    print "              with this CMSSW version of PAT                         "
    print "              it is required to re-run the GenJet production for     "
    print "              anti-kT since that is not part of the re-reco          "
    print "*********************************************************************"
    process.load("RecoJets.Configuration.GenJetParticles_cff")
    process.load("RecoJets.JetProducers.ak5GenJets_cfi")
    process.ak7GenJets = process.ak5GenJets.clone( rParam = 0.7 )
    process.makePatJets.replace( process.patJetCharge, process.genParticlesForJets+process.ak5GenJets+process.ak7GenJets+process.patJetCharge)
    #-- Remove changes for GenJets ------------------------------------------------
    process.genParticlesForJets.ignoreParticleIDs = cms.vuint32(1000022, 1000012, 1000014, 1000016, 2000012,
        2000014, 2000016, 1000039, 5100039, 4000012,
        4000014, 4000016, 9900012, 9900014, 9900016,
        39)
    process.genParticlesForJets.excludeResonances = True



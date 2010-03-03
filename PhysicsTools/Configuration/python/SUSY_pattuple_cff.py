#
#  SUSY-PAT configuration fragment
#
#  PAT configuration for the SUSY group - 35X series
#  More information here:
#  https://twiki.cern.ch/twiki/bin/view/CMS/SusyPatLayer1DefV8


import FWCore.ParameterSet.Config as cms

def addDefaultSUSYPAT(process, mcInfo=True, HLTMenu='HLT', JetMetCorrections='Summer09_7TeV_ReReco332',theJetNames = ['IC5Calo','SC5Calo','AK5PF','AK5JPT','AK5Track']):
    if not mcInfo:
	removeMCDependence(process)
    loadPAT(process,JetMetCorrections)
    addJetMET(process,theJetNames)
    loadPATTriggers(process,HLTMenu)
    #not included for the time being
    #loadPF2PAT(process,mcInfo)

    #-- Counter for the number of processed events --------------------------------
    process.eventCountProducer = cms.EDProducer("EventCountProducer")

    # Full path
    process.seqSUSYDefaultSequence = cms.Sequence( process.jpt * process.addTrackJets
                                                   * process.patDefaultSequence #* process.patDefaultSequencePF
                                                   * process.patTrigger*process.patTriggerEvent
						                           * process.eventCountProducer )

def loadPAT(process,JetMetCorrections):
    #-- PAT standard config -------------------------------------------------------
    process.load("PhysicsTools.PatAlgos.patSequences_cff")
    #-- Changes for electron and photon ID ----------------------------------------
    # Turn off photon-electron cleaning (i.e., flag only)
    process.cleanPatPhotons.checkOverlaps.electrons.requireNoOverlaps = False

    # Remove embedding of superClusters, will keep entire superCluster collection
    process.patElectrons.embedSuperCluster = False
    process.patPhotons.embedSuperCluster   = False
    
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

    #-- Jet corrections -----------------------------------------------------------
    process.patJetCorrFactors.corrSample = JetMetCorrections 

def loadPF2PAT(process,mcInfo): 
    #-- PF2PAT config -------------------------------------------------------------
    #process.load("PhysicsTools.PatAlgos.patSequences_cff")
    #from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
    #cloneProcessingSnippet(process, process.patDefaultSequence, "PF")
    from PhysicsTools.PatAlgos.tools.pfTools import usePF2PAT
    usePF2PAT(process,runPF2PAT=True, jetAlgo='AK5',runOnMC=mcInfo)
    
    #process.load("RecoTauTag.RecoTau.PFRecoTauDiscriminationLowPt_cff")
    #process.patDefaultSequence.replace(process.pfTauSequence,process.pfTauSequence + process.TauDiscrForLowPt)
    #process.pfLayer1Taus.tauIDSources.LowPtTausDiscr=cms.InputTag("DiscrLowPtTau")
    #process.pfMuonsPtGt5.ptMin = cms.double(2.0)
    #process.pfElectronsPtGt5.ptMin = cms.double(2.0)
	
    
def loadPATTriggers(process,HLTMenu):
    #-- Trigger matching ----------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
    switchOnTrigger( process )
    process.patTriggerSequence.remove( process.patTriggerMatcher )
    process.patTriggerEvent.patTriggerMatches  = ()
    # If we have to rename the default trigger menu
    process.patTrigger.processName = HLTMenu
    process.patTriggerEvent.processName = HLTMenu

def addSUSYJetCollection(process,jets = 'IC5Calo',doJTA=False,doType1MET=False,doJetID = True,jetIdLabel = None):
    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection, addJetID
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
    elif type == 'JPT':
        if 'IC' in algorithm: collectionJPT = algorithm.replace('IC','Icone')
        elif 'SC' in algorithm: collectionJPT = algorithm.replace('SC','Siscone')
        elif 'AK' in algorithm: collectionJPT = algorithm.replace('AK','AntiKt')
        else: raise ValueError, "Unknown jet algorithm: %s" % (jets)
        jetCollection = 'JetPlusTrackZSPCorJet%(collectionJPT)s' % locals()
    	jetCorrLabel = None
        jetIdLabel =  '%(collection)sJPT' % locals()
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

def addJetMET(process,theJetNames):
    
    #-- Jet plus tracks -----------------------------------------------------------
    process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")
    process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
    
    process.jptCaloJets = cms.Sequence(
    	process.ZSPJetCorrectionsIcone5 *
    	process.ZSPJetCorrectionsSisCone5 *
    	process.ZSPJetCorrectionsAntiKt5 *
    	process.JetPlusTrackCorrectionsIcone5 *
    	process.JetPlusTrackCorrectionsSisCone5 *
    	process.JetPlusTrackCorrectionsAntiKt5
    )

    process.load("JetMETCorrections.JetPlusTrack.matchJptAndCaloJets_cff")
    process.load("JetMETCorrections.JetPlusTrack.jptJetId_cff")
    process.jpt = cms.Sequence( process.jptCaloJets * process.matchJptAndCaloJets * process.jptJetId)
    
    #-- Track Jets ----------------------------------------------------------------
    process.load('RecoJets.Configuration.RecoTrackJets_cff')
    process.addTrackJets = cms.Sequence ( process.recoTrackJets )
 
    #-- Extra Jet/MET collections -------------------------------------------------
    # Add a few jet collections...
    for jetName in theJetNames:
    	addSUSYJetCollection(process,jetName)
    
    #-- Tune contents of jet collections  -----------------------------------------
    theJetNames.append('')
    for jetName in theJetNames:
        module = getattr(process,'patJets'+jetName)
        module.addTagInfos = False    # Remove tag infos
        module.embedGenJetMatch = False # Only keep reference, since we anyway keep the genJet collections
    theJetNames.pop()
    
    # Add tcMET, pfMET 
    from PhysicsTools.PatAlgos.tools.metTools import addTcMET, addPfMET
    addTcMET(process,'TC')
    addPfMET(process,'PF')

    # Rename default jet collection for uniformity
    process.cleanPatJetsAK5Calo = process.cleanPatJets
    process.patMETsAK5Calo      = process.patMETs
    process.patMHTsAK5Calo      = process.patMHTs

    # Modify subsequent modules
    process.patHemispheres.patJets = process.cleanPatJetsAK5Calo.label()
    process.countPatJets.src       = process.cleanPatJetsAK5Calo.label()
    
    # Add MHT (inserted until officially suported)
    from PhysicsTools.PatAlgos.producersLayer1.mhtProducer_cff import makePatMHTs, patMHTs
    process.countPatCandidates.replace(process.countPatJets, process.countPatJets + process.makePatMHTs)
    process.patMHTs.jetTag      = 'patJets'
    process.patMHTs.electronTag = 'patElectrons'
    process.patMHTs.muonTag     = 'patMuons'
    process.patMHTs.tauTag      = 'patTaus'
    process.patMHTs.photonTag   = 'patPhotons'

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

def getSUSY_pattuple_outputCommands( process ):
    return [ # PAT Objects
        'keep *_cleanPatPhotons_*_*',
        'keep *_cleanPatElectrons_*_*',
        'keep *_cleanPatMuons_*_*',
        'keep *_cleanPatTaus_*_*',
        'keep *_cleanPatJets*_*_*',       # All Jets
        'keep *_patMETs*_*_*',            # All METs
        'keep *_patMHTs*_*_*',            # All MHTs
        'keep *_cleanPatHemispheres_*_*',
        'keep *_cleanPatPFParticles_*_*',
        # Generator information
        'keep GenEventInfoProduct_generator_*_*',
        'keep GenRunInfoProduct_generator_*_*',
        # Generator particles/jets/MET
        'keep recoGenParticles_genParticles_*_*',
        'keep recoGenJets_*GenJets*_*_*',
        'keep recoGenMETs_*_*_*',
        # Trigger information
        'keep edmTriggerResults_TriggerResults_*_HLT*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep L1GlobalTriggerObjectMapRecord_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_*_*_*',
        # Others
        'keep recoCaloMET_met_*_*', # raw MET
        'keep *_muon*METValueMapProducer_*_*',   # Muon corrections to MET
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_towerMaker_*_*',                 # Keep CaloTowers for cross-cleaning
        'keep edmMergeableCounter_eventCountProducer_*_*',
        'keep recoTracks_generalTracks_*_*',
	'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*',
	'keep recoTrackJets_ak5TrackJets_*_*',
	'keep *_pfLayer*_*_*', # Keep PF2PAT output
	'keep *_electronMergedSeeds_*_*',
	'keep *_Conversions_*_*',
        'drop patPFParticles_pfLayer*_*_*', # drop PAT particles
	'keep recoPFCandidates_particleFlow_*_*',
        'keep recoSuperClusters_corrected*_*_*',
	'keep recoSuperClusters_pfElectronTranslator_*_*',
        'keep *_gsfElectronCores_*_*',    #Keep electron core
        'keep *_photonCore_*_*',        #Keep electron core
        'keep recoConversions_conversions_*_*',
        'keep recoTracks_*onversions_*_*',
        'keep HcalNoiseSummary_*_*_*' #Keep the one in RECO
        ] 

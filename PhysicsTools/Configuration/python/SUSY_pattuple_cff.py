#
#  SUSY-PAT configuration fragment
#
#  PAT configuration for the SUSY group - 35X/36X series
#  More information here:
#  https://twiki.cern.ch/twiki/bin/view/CMS/SusyPatLayer1DefV8


import FWCore.ParameterSet.Config as cms

def addDefaultSUSYPAT(process, mcInfo=True, HLTMenu='HLT', JetMetCorrections='Summer09_7TeV_ReReco332', mcVersion='' ,theJetNames = ['IC5Calo','IC5PF','AK5JPT','AK5Track']):
    loadPF2PAT(process,mcInfo,'PF')
    if not mcInfo:
	removeMCDependence(process)
    loadMCVersion(process,mcVersion,mcInfo)
    loadPAT(process,JetMetCorrections)
    addJetMET(process,theJetNames)
    loadPATTriggers(process,HLTMenu)

    #-- Counter for the number of processed events --------------------------------
    process.eventCountProducer = cms.EDProducer("EventCountProducer")

    # Full path
    process.susyPatDefaultSequence = cms.Sequence( process.eventCountProducer 
                                                   * process.patDefaultSequence * process.patPF2PATSequencePF
                                                   * process.patTrigger * process.patTriggerEvent
                                                    )

    if mcVersion == '35x' and 'JPT' in ''.join(theJetNames): 
    	process.susyPatDefaultSequence.replace(process.eventCountProducer, process.eventCountProducer * process.recoJPTJets)

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

def loadPF2PAT(process,mcInfo,postfix):
    #-- PF2PAT config -------------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.pfTools import usePF2PAT
    usePF2PAT(process,runPF2PAT=True, jetAlgo='AK5',runOnMC=mcInfo,postfix=postfix)

def loadPATTriggers(process,HLTMenu):
    #-- Trigger matching ----------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
    switchOnTrigger( process )
    process.patTriggerSequence.remove( process.patTriggerMatcher )
    process.patTriggerEvent.patTriggerMatches  = []
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
    	#jetCorrLabel = None
	#doJetID = False
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
    
    # Add tcMET
    from PhysicsTools.PatAlgos.tools.metTools import addTcMET #, addPfMET
    addTcMET(process,'TC')
    #addPfMET(process,'PF') #is in PF2PAT

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
	# Keep PF2PAT output
        'keep *_selectedPatMuonsPF_*_*',         
        'keep *_selectedPatElectronsPF_*_*',         
        'keep *_selectedPatTausPF_*_*',         
        'keep *_selectedPatJetsPF_*_*',         
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
        #Pat trigger matching
        'keep patTriggerObjects_patTrigger_*_*',
        'keep patTriggerFilters_patTrigger_*_*',
        'keep patTriggerPaths_patTrigger_*_*',
        'keep patTriggerEvent_patTriggerEvent_*_*',
        'keep *_cleanPatPhotonsTriggerMatch_*_*',
        'keep *_cleanPatElectronsTriggerMatch_*_*',
        'keep *_cleanPatMuonsTriggerMatch_*_*',
        'keep *_cleanPatTausTriggerMatch_*_*',
        'keep *_cleanPatJetsTriggerMatch_*_*',
        'keep *_patMETsTriggerMatch_*_*',
        'keep patTriggerObjectStandAlones_patTrigger_*_*',
        'keep patTriggerObjectStandAlonesedmAssociation_*_*_*',
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

def run36xOnReRecoMC( process, genJets = "ak5GenJets"):
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
    process.load("RecoJets.JetProducers." + genJets +"_cfi")
    process.makePatJets.replace( process.patJetCharge, process.genParticlesForJets+getattr(process,genJets)+process.patJetCharge)


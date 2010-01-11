#
#  SUSY-PAT configuration fragment
#
#  PAT configuration for the SUSY group - 33X/34X series
#  More information here:
#  https://twiki.cern.ch/twiki/bin/view/CMS/SusyPatLayer1DefV7


import FWCore.ParameterSet.Config as cms

def addDefaultSUSYPAT(process, mcInfo=True, HLTMenu='HLT', JetMetCorrections='Summer09_7TeV',mcVersion=''):
    
    if not mcInfo:
	removeMCDependence(process)
    else:
	loadMCVersion(process,mcVersion)
    loadPAT(process,JetMetCorrections)
    addJetMET(process)
    loadPATTriggers(process,HLTMenu)
    loadPF2PAT(process,mcInfo)
    if mcVersion == '31x' and mcInfo:
	runSUSY33xOn31xMC(process)

    # Full path
    process.seqSUSYDefaultSequence = cms.Sequence( process.jpt * process.addTrackJets
                                                   *process.patDefaultSequence
                                                   * process.patTrigger*process.patTriggerEvent
						   * process.PFPATafterPAT )

def loadMCVersion(process, mcVersion):
    #-- Missing ak5GenJets in 3.3.2 samples ---------------------------------------
    from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run33xOnReRecoMC, run33xOn31xMC
    if mcVersion == '31x':
    	run33xOn31xMC( process )
    if mcVersion == '31xReReco332':
	run33xOnReRecoMC( process, "ak5GenJets" )


def loadPAT(process,JetMetCorrections):
    #-- PAT standard config -------------------------------------------------------
    process.load("PhysicsTools.PatAlgos.patSequences_cff")

    #-- Changes for electron and photon ID ----------------------------------------
    # Turn off photon-electron cleaning (i.e., flag only)
    process.cleanLayer1Photons.checkOverlaps.electrons.requireNoOverlaps = False

    # Remove embedding of superClusters, will keep entire superCluster collection
    process.allLayer1Electrons.embedSuperCluster = False
    process.allLayer1Photons.embedSuperCluster   = False
    
    #-- Tuning of Monte Carlo matching --------------------------------------------
    # Also match with leptons of opposite charge
    process.electronMatch.checkCharge = False
    process.muonMatch.checkCharge     = False
    process.tauMatch.checkCharge      = False
    process.tauMatch.maxDeltaR        = cms.double(0.3)

    #-- Jet corrections -----------------------------------------------------------
    process.jetCorrFactors.corrSample = JetMetCorrections ## 'Summer09' for 10TeV, 'Summer09_7TeV' for 7TeV no ReReco

def loadPF2PAT(process,mcInfo):
    #-- PF2PAT config -------------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.pfTools import usePATandPF2PAT, removeMCDependencedorPF
    usePATandPF2PAT(process,runPATandPF2PAT=True, jetAlgo='IC5')
    if not mcInfo:
	removeMCDependencedorPF(process)
    
    process.load("RecoTauTag.RecoTau.PFRecoTauDiscriminationLowPt_cff")
    process.PFPATafterPAT.replace(process.pfTauSequence,process.pfTauSequence + process.TauDiscrForLowPt)
    process.pfLayer1Taus.tauIDSources.LowPtTausDiscr=cms.InputTag("DiscrLowPtTau")
    process.pfMuonsPtGt5.ptMin = cms.double(2.0)
    process.pfElectronsPtGt5.ptMin = cms.double(2.0)
	
    
def loadPATTriggers(process,HLTMenu):
    #-- Trigger matching ----------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
    switchOnTrigger( process )
    process.patTriggerSequence.remove( process.patTriggerMatcher )
    process.patTriggerEvent.patTriggerMatches  = ()
    # If we have to rename the default trigger menu
    process.patTrigger.processName = HLTMenu
    process.patTriggerEvent.processName = HLTMenu

def addJetMET(process):
    #-- Extra Jet/MET collections -------------------------------------------------
    # Add a few jet collections...
    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    
    #-- Jet plus tracks -----------------------------------------------------------
    process.load("PhysicsTools.PatAlgos.recoLayer0.jetPlusTrack_cff")
    process.jpt = cms.Sequence( process.jptCaloJets )
	
    # CaloJets
    addJetCollection(process, cms.InputTag('iterativeCone5CaloJets'),
                     'IC5',
                     doJTA            = True,
                     doBTagging       = True,
                     jetCorrLabel     = ('IC5','Calo'),
                     doType1MET       = True,
                     doL1Cleaning     = True,
                     doL1Counters     = True,
                     doJetID          = True,
		     jetIdLabel       = "ic5",
                     genJetCollection = cms.InputTag("iterativeCone5GenJets")
                     )
    addJetCollection(process,cms.InputTag('sisCone5CaloJets'),
                     'SC5',
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = ('SC5','Calo'),
                     doType1MET   = True,
                     doL1Cleaning = True,
                     doL1Counters = True,
                     doJetID      = True,
                     jetIdLabel   = "sc5",
                     genJetCollection=cms.InputTag("sisCone5GenJets")
                     )
    # PF jets
    addJetCollection(process,cms.InputTag('ak5PFJets'),
                     'AK5PF',
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = ('AK5','PF'),
                     doType1MET   = False,
                     doL1Cleaning = True,
                     doL1Counters = True,
                     doJetID      = False,
                     genJetCollection=cms.InputTag("ak5GenJets")
                     )
    addJetCollection(process,cms.InputTag('sisCone5PFJets'),
                     'SC5PF',
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = ('SC5','PF'),
                     doType1MET   = False,
                     doL1Cleaning = True,
                     doL1Counters = True,
                     doJetID      = False,
                     genJetCollection=cms.InputTag("sisCone5GenJets")
                     )
    
    # JPT jets
    addJetCollection(process,cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
                     'AK5JPT',
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = None,
                     doType1MET   = False,
                     doL1Cleaning = True,
                     doL1Counters = True,
                     doJetID      = False,
                     genJetCollection = cms.InputTag("ak5GenJets")
                     )
	
    #-- Track Jets ----------------------------------------------------------------
    process.load('RecoJets.Configuration.RecoTrackJets_cff')
    process.addTrackJets = cms.Sequence ( process.recoTrackJets )
    addJetCollection(process,cms.InputTag('ak5TrackJets'),
                     'AK5Track',
                     doJTA        = False,
                     doBTagging   = True,
                     jetCorrLabel = None,
                     doType1MET   = False,
                     doL1Cleaning = True,
                     doL1Counters = True,
                     doJetID      = False,
                     genJetCollection = cms.InputTag("ak5GenJets")
                     )
    
    #-- Tune contents of jet collections  -----------------------------------------
    for jetName in ( '', 'IC5', 'SC5' , 'AK5PF', 'SC5PF', 'AK5JPT', 'AK5Track'):
        module = getattr(process,'allLayer1Jets'+jetName)
        module.addTagInfos = False    # Remove tag infos
        module.embedGenJetMatch = False # Only keep reference, since we anyway keep the genJet collections
 
    # Add tcMET 
    from PhysicsTools.PatAlgos.tools.metTools import addTcMET #, addPfMET
    addTcMET(process,'TC')
    #addPfMET(process,'PF')

    # Rename default jet collection for uniformity
    process.cleanLayer1JetsAK5 = process.cleanLayer1Jets
    process.layer1METsAK5      = process.layer1METs
    process.layer1MHTsAK5      = process.layer1MHTs

    # Modify subsequent modules
    process.cleanLayer1Hemispheres.patJets = process.cleanLayer1JetsAK5.label()
    process.countLayer1Jets.src            = process.cleanLayer1JetsAK5.label()

    # Modify counters' input
    process.allLayer1Summary.candidates.remove(cms.InputTag('layer1METs'))
    process.allLayer1Summary.candidates.append(cms.InputTag('layer1METsAK5'))
    process.allLayer1Summary.candidates.remove(cms.InputTag('layer1MHTs'))
    process.allLayer1Summary.candidates.append(cms.InputTag('layer1MHTsAK5'))
    process.cleanLayer1Summary.candidates.remove(cms.InputTag('cleanLayer1Jets'))
    process.cleanLayer1Summary.candidates.append(cms.InputTag('cleanLayer1JetsAK5'))
    # Add new jet collections to counters (MET done automatically)
    for jets in ( 'IC5', 'SC5', 'AK5PF', 'SC5PF', 'AK5JPT', 'AK5Track'): 
        process.allLayer1Summary.candidates.append(cms.InputTag('allLayer1Jets'+jets))
        process.selectedLayer1Summary.candidates.append(cms.InputTag('selectedLayer1Jets'+jets))
        process.cleanLayer1Summary.candidates.append(cms.InputTag('cleanLayer1Jets'+jets))
	

def removeMCDependence( process ):
    #-- Remove MC dependence ------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.coreTools import removeMCMatching
    removeMCMatching(process, 'All')

def getSUSY_pattuple_outputCommands( process ):
    return [ # PAT Objects
        'keep *_cleanLayer1Photons_*_*',
        'keep *_cleanLayer1Electrons_*_*',
        'keep *_cleanLayer1Muons_*_*',
        'keep *_cleanLayer1Taus_*_*',
        'keep *_cleanLayer1Jets*_*_*',       # All Jets
        'keep *_layer1METs*_*_*',            # All METs
        'keep *_layer1MHTs*_*_*',            # All MHTs
        'keep *_cleanLayer1Hemispheres_*_*',
        'keep *_cleanLayer1PFParticles_*_*',
        # Generator information
        'keep GenEventInfoProduct_generator_*_*',
        'keep GenRunInfoProduct_generator_*_*',
        # Generator particles/jets/MET
        'keep recoGenParticles_genParticles_*_*',
        'keep recoGenJets_iterativeCone5GenJets_*_*',
        'keep recoGenJets_sisCone5GenJets_*_*',
        'keep recoGenJets_ak5GenJets*_*_*',
        'keep recoGenMETs_*_*_*',
        # Trigger information
        'keep edmTriggerResults_TriggerResults_*_HLT',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep L1GlobalTriggerObjectMapRecord_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_*_*_*',
        # Others
        'keep *_muon*METValueMapProducer_*_*',   # Muon corrections to MET
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_towerMaker_*_*',                 # Keep CaloTowers for cross-cleaning
        'keep edmMergeableCounter_eventCountProducer_*_*',
        'keep recoTracks_generalTracks_*_*',
	#'keep recoTrackExtras_*_*_*',
	'keep *_pfLayer*_*_*', # Keep PF2PAT output
	'keep *_electronMergedSeeds_*_*',
	'keep *_Conversions_*_*',
	#'keep recoGsfTracks_electronGsfTracks_*_*',
	#'keep recoTracks_standAloneMuons_*_*',
	#'keep recoTracks_globalMuons_*_*',
	#'keep *_muons_*_*',
	#'keep *_gsfElectrons_*_*',
	#'keep *_softPFElectrons_*_*',
	#'keep *_eid*_*_*',
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

def runSUSY33xOn31xMC(process):
   process.ZSPJetCorJetAntiKt5.src = "antikt5CaloJets"
   process.jetGenJetMatchAK5Track.matched = "antikt5GenJets"
   process.jetGenJetMatchAK5JPT.matched = "antikt5GenJets"
   process.jetGenJetMatchAK5Track.matched = "antikt5GenJets"
   process.jetGenJetMatchAK5PF.matched = "antikt5GenJets"

   process.jetTracksAssociatorAtVertexAK5PF.jets = "antikt5PFJets"
   process.jetGenJetMatchAK5Track.src = "antikt5GenJets"

   from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
   process.ak5PFJets = ak5PFJets

   from PhysicsTools.PatAlgos.tools.cmsswVersionTools import addJetID
   #addJetID("antikt5CaloJets", "ak5JetID")
   addJetID(process, "iterativeCone5CaloJets","ic5")
   addJetID(process, "sisCone5CaloJets","sc5")

   process.seqAdditionalRECO = cms.Sequence( process.ak5PFJets )

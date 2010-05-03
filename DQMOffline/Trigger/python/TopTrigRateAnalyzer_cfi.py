import FWCore.ParameterSet.Config as cms

topTrigOfflineDQM = cms.EDAnalyzer("TopTrigAnalyzer",

    HltProcessName = cms.string("HLT"),

    createStandAloneHistos = cms.untracked.bool(False),
	histoFileName = cms.untracked.string("MuonTrigHistos_RECO.root"),

   # The muon collection tag must point to a
   # a group of muon-type objects, not just track objects									
   RecoMuonInputTag = cms.InputTag("muons", "", ""),
   BeamSpotInputTag = cms.InputTag("offlineBeamSpot", "", ""),
   HltRawInputTag = cms.InputTag("hltTriggerSummaryRAW", "", ""),
   HltAodInputTag = cms.InputTag("hltTriggerSummaryAOD", "", ""),
   # This is still used
   # to select based on trigger
   TriggerResultLabel = cms.InputTag("TriggerResults","","HLT"),
   
   #### jet selection
   CaloJetInputTag = cms.InputTag("iterativeCone5CaloJets","",""),
   
   EtaCut = cms.untracked.double(2.4),
   PtCut  = cms.untracked.double(8.0),
   NJets  = cms.untracked.int32(2),

									
   # Define the cuts for your muon selections
  customCollection = cms.VPSet(


	cms.untracked.PSet(
	  collectionName = cms.untracked.string ("topMuonPt1_anyTrig"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0), # 3 meters
	  chi2cut = cms.untracked.double(30.0),
	  nHits = cms.untracked.int32(20),
	  recoCuts = cms.untracked.string ("pt > 1 && abs(eta) < 2.1"),
	  hltCuts  = cms.untracked.string ("pt > 1 && abs(eta) < 2.1")
	),

	

	cms.untracked.PSet(
	#  collectionName = cms.untracked.string ("topMuonPt15_QuadJet15U"),
	  collectionName = cms.untracked.string ("topMuonPt1_QuadJet15U"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	#  requiredTriggers = cms.untracked.vstring("HLT_QuadJet15U"),
	  requiredTriggers = cms.untracked.vstring("HLT_QuadJet15U"),
	  d0cut = cms.untracked.double(0.2),
	  z0cut = cms.untracked.double(25.0),
	  chi2cut = cms.untracked.double(30.0),
	  nHits = cms.untracked.int32(20),
	  recoCuts = cms.untracked.string ("pt > 1 && abs(eta) < 2.1"),
	  hltCuts  = cms.untracked.string ("pt > 1 && abs(eta) < 2.1")	  
	),



	cms.untracked.PSet(
	#  collectionName = cms.untracked.string ("topMuonPt15_QuadJet15U"),
	  collectionName = cms.untracked.string ("topMuonPt1_QuadJet30"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	#  requiredTriggers = cms.untracked.vstring("HLT_QuadJet15U"),
	  requiredTriggers = cms.untracked.vstring("HLT_QuadJet30"),
	  d0cut = cms.untracked.double(0.2),
	  z0cut = cms.untracked.double(25.0),
	  chi2cut = cms.untracked.double(30.0),
	  nHits = cms.untracked.int32(20),
	  recoCuts = cms.untracked.string ("pt > 1 && abs(eta) < 2.1"),
	  hltCuts  = cms.untracked.string ("pt > 1 && abs(eta) < 2.1")	  
	),
	
	
									  
    ),

    # Set the ranges and numbers of bins for histograms
	# max pt is not very useful


    #EtaParameters      = cms.vdouble(50, -3.5,3.5),

    EtaParameters      = cms.untracked.vdouble(20, -2.1,2.1),
    PhiParameters      = cms.untracked.vdouble(20, -3.15,3.15),
    ResParameters      = cms.untracked.vdouble(20, -0.15, 0.15),
    DrParameters       = cms.untracked.vdouble(20, 0.0, 0.05),
	
    JetMParameters     = cms.untracked.vdouble(11, -0.5, 10.5),			

    # Use Pt Parameters to set bin edges

    PtParameters       = cms.untracked.vdouble(0.0,  2.0,  4.0, 
											   6.0, 8.0, 10.0,
											   20.0, 30.0, 40.0,
											   100.0, 200.0, 400.0),

    Z0Parameters       = cms.untracked.vdouble(25, -25, 25),
    D0Parameters       = cms.untracked.vdouble(25, -1, 1),									




	# valid match types are dr and cosmic
	# future update: make sure default is
	# delta r matching
	matchType = cms.untracked.string("dr"),

    # If you have cosmic matching
	# you will ignore the delta R cuts								
								   									
	L1DrCut   = cms.untracked.double(0.4),
	L2DrCut   = cms.untracked.double(0.25),
	L3DrCut   = cms.untracked.double(0.025),								
									

    DQMStore = cms.untracked.bool(True),

	# still used by overlap analyzer
    # not included in meaningful output									
    CrossSection = cms.double(0.97),
	# ddd 								
    #NSigmas90 = cms.untracked.vdouble(3.0, 3.0, 3.0, 3.0),


	# list of triggers
    # any triggers not in the hlt configuraiton
    # will be ignored
									
	TriggerNames = cms.vstring(
	    "HLT_L1Mu",
		"HLT_L1MuOpen",
		"HLT_Mu3",
		"HLT_Mu5",
        "HLT_IsoMu3",
        "HLT_Mu5",
        "HLT_Mu9",        
    ),


)

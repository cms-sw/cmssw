import FWCore.ParameterSet.Config as cms

offlineDQMMuonTrig = cms.EDAnalyzer("OfflineDQMMuonTrigAnalyzer",

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

									
   # Define the cuts for your muon selections
   customCollection = cms.VPSet(

 	cms.untracked.PSet(
 	  collectionName = cms.untracked.string ("dqmBarrelMuons"),
 	  # The track collection is just a switch, not a true collection name
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring("HLT_L1MuOpen"),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0),
	  recoCuts = cms.untracked.string ("abs(eta) < 0.9"),
	  hltCuts  = cms.untracked.string ("abs(eta) < 0.9")	  
	),


	cms.untracked.PSet(
	  collectionName = cms.untracked.string ("endcapMuons"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring("HLT_L1MuOpen"),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0), # 3 meters
	  recoCuts = cms.untracked.string ("abs(eta) > 1.4 && abs(eta) < 2.0"),
	  hltCuts  = cms.untracked.string ("abs(eta) > 1.4 && abs(eta) < 2.0")
	),

	cms.untracked.PSet(
	  collectionName = cms.untracked.string ("allMuons"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring("HLT_L1MuOpen"),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0),
	  recoCuts = cms.untracked.string ("abs(eta) < 2.0"),
	  hltCuts  = cms.untracked.string ("abs(eta) < 2.0")	  
	),
									  
    ),

    # Set the ranges and numbers of bins for histograms
	# max pt is not very useful
    MaxPtParameters    = cms.vdouble(25,0.,100.),
    # PtParmeters is not currently used

    EtaParameters      = cms.vdouble(25, -3.5,3.5),
    PhiParameters      = cms.vdouble(25, -3.15,3.15),
    ResParameters      = cms.vdouble(25, -0.15, 0.15),
	DrParameters       = cms.vdouble(25, 0.0, 0.05),			

    # Use Pt Parameters to set bin edges

    PtParameters       = cms.vdouble(0.0,  2.0,  4.0, 
									 6.0, 8.0, 10.0, 
									 12.0,  14.0,  16.0, 
									 18.0,  20.0,
									 22.0, 24.0, 26.0, 28.0,
									 30.0, 32.0, 34.0, 36.0,
									 38.0, 40.0,
									 45.0, 50.0, 55.0, 60.0,
									 65.0, 70.0, 75.0, 80.0,
									 85.0, 90.0, 95.0, 100.0,
									 125.0, 150.0, 175.0, 200.0,
									 300.0),

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
		"HLT_L1Mu20",
        "HLT_L2Mu9",
		"HLT_L2Mu11",
        "HLT_IsoMu3",        
        "HLT_Mu3",
        "HLT_Mu5",
        "HLT_Mu9",        
        "HLT_DoubleMu3",
        "HLT_DoubleMu0",
		"HLT_L1DoubleMuOpen"		
    ),


)

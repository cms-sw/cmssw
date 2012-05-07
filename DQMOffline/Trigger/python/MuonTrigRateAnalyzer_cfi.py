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
 	  collectionName = cms.untracked.string ("allStandAloneMuons"),
 	  # The track collection is just a switch, not a true collection name
	  # track collection can be innerTrack, outerTrack, globalTrack
	  # innerTrack means the muon will required to be a tracker muon
	  # outerTrack means that muon will required to be a standalone muon
	  # globalTrack means that the muon will be required to be a global muon
	  trackCollection = cms.untracked.string ("outerTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0),
	  recoCuts = cms.untracked.string ("abs(eta) < 2.0"),
	  hltCuts  = cms.untracked.string ("abs(eta) < 2.0")	  
	),




 	cms.untracked.PSet(
 	  collectionName = cms.untracked.string ("allTrackerMuons"),
 	  # The track collection is just a switch, not a true collection name
	  # track collection can be innerTrack, outerTrack, globalTrack
	  # innerTrack means the muon will required to be a tracker muon
	  # outerTrack means that muon will required to be a standalone muon
	  # globalTrack means that the muon will be required to be a global muon

	  trackCollection = cms.untracked.string ("innerTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0),
	  recoCuts = cms.untracked.string ("abs(eta) < 2.0"),
	  hltCuts  = cms.untracked.string ("abs(eta) < 2.0")	  
	),


	

	cms.untracked.PSet(
	  collectionName = cms.untracked.string ("allMuons"),
	  # 
	  # innerTrack means the muon will required to be a tracker muon
	  # outerTrack means that muon will required to be a standalone muon
	  # globalTrack means that the muon will be required to be a global muon
      #
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0),
	  recoCuts = cms.untracked.string ("abs(eta) < 2.0"),
	  hltCuts  = cms.untracked.string ("abs(eta) < 2.0")	  
	),
									  
    ),

    # Set the ranges and numbers of bins for histograms
	# max pt is not very useful
    #MaxPtParameters    = cms.vdouble(25,0.,100.),
    # PtParmeters is not currently used

    EtaParameters      = cms.untracked.vdouble(20, -2.1,2.1),
    PhiParameters      = cms.untracked.vdouble(20, -3.15,3.15),
    ResParameters      = cms.untracked.vdouble(20, -0.15, 0.15),
	DrParameters       = cms.untracked.vdouble(20, 0.0, 0.1),			

    # Use Pt Parameters to set bin edges

    PtParameters       = cms.untracked.vdouble(0.0,  2.0,  4.0, 
									 6.0, 8.0, 10.0, 
									 20.0,  30.0,  40.0, 
									 100.0,   200.0,
									 400.0),

    Z0Parameters       = cms.untracked.vdouble(10, -15, 15),
    D0Parameters       = cms.untracked.vdouble(10, -0.5, 0.5),									

	# valid match types are dr and cosmic
	# future update: make sure default is
	# delta r matching
	matchType = cms.untracked.string("dr"),

    RequireRecoToMatchL1Seed = cms.untracked.bool(True),									

    # If you have cosmic matching
	# you will ignore the delta R cuts								
								   									
	L1DrCut   = cms.untracked.double(0.4),
	L2DrCut   = cms.untracked.double(0.3),
	L3DrCut   = cms.untracked.double(0.05),								
									

    DQMStore = cms.untracked.bool(True),

	# still used by overlap analyzer
    # not included in meaningful output									
    #CrossSection = cms.double(0.97),
	# ddd 								
    #NSigmas90 = cms.untracked.vdouble(3.0, 3.0, 3.0, 3.0),


	# list of triggers
    # any triggers not in the hlt configuraiton
    # will be ignored
	# This list contains triggers from both 8E29, 1E31
    # 									
									

    TriggerRegExpStrings = cms.vstring(
                       "HLT_(HI)?(L[12])?(Single)?(Double)?(Iso)?Mu[0-9]*(Open)?(_NoVertex)?(_Core)?(_v[0-9]*)?$",
                             )

)

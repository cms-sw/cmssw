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
	  collectionName = cms.untracked.string ("dqmBarrelMuons3"),
	  # The track collection is just a switch, not a true collection name
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0),
	  recoCuts = cms.untracked.string ("pt > 3 && abs(eta) < 0.9"),
	  hltCuts  = cms.untracked.string ("pt > 3 && abs(eta) < 0.9")	  
	),

	cms.untracked.PSet(
	  collectionName = cms.untracked.string ("endcapMuons3"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
	  d0cut = cms.untracked.double(3.0),
	  z0cut = cms.untracked.double(25.0), # 3 meters
	  recoCuts = cms.untracked.string ("pt > 3 && abs(eta) > 1.4 && abs(eta) < 2.4"),
	  hltCuts  = cms.untracked.string ("pt > 3 && abs(eta) > 1.4 && abs(eta) < 2.4")
	),

	cms.untracked.PSet(
	  collectionName = cms.untracked.string ("allMuonPt3"),
	  trackCollection = cms.untracked.string ("globalTrack"),
	  requiredTriggers = cms.untracked.vstring("HLT_QuadJet15U"),
	  d0cut = cms.untracked.double(2.0),
	  z0cut = cms.untracked.double(25.0),
	  recoCuts = cms.untracked.string ("pt > 3 "),
	  hltCuts  = cms.untracked.string ("pt > 3 ")	  
	),
									  
    ),

    # Set the ranges and numbers of bins for histograms
	# max pt is not very useful
    MaxPtParameters    = cms.vdouble(40,0.,80.),
    # PtParmeters is not currently used
	PtParameters       = cms.vdouble(50,0.,80.),
    EtaParameters      = cms.vdouble(50, -3.5,3.5),
    PhiParameters      = cms.vdouble(50, -3.15,3.15),
    ResParameters      = cms.vdouble(50, -0.15, 0.15),
	DrParameters       = cms.vdouble(50, 0.0, 0.05),			


	# valid match types are dr and cosmic
	# future update: make sure default is
	# delta r matching
	matchType = cms.untracked.string("dr"),

    # If you have cosmic matching
	# you will ignore the delta R cuts								
								   									
	L1DrCut   = cms.untracked.double(0.4),
	L2DrCut   = cms.untracked.double(0.25),
	L3DrCut   = cms.untracked.double(0.015),								
									

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
        "HLT_L2Mu9",
        "HLT_IsoMu9",
        "HLT_IsoMu11",
        "HLT_IsoMu13",
        "HLT_IsoMu15",
        "HLT_Mu3",
        "HLT_Mu5",
        "HLT_Mu7",
        "HLT_Mu9",
        "HLT_Mu11",
        "HLT_Mu13",
        "HLT_Mu15",
        "HLT_Mu15_L1Mu7",
        "HLT_Mu15_Vtx2cm",
        "HLT_Mu15_Vtx2mm",
        "HLT_DoubleIsoMu3",
        "HLT_DoubleMu3",
        "HLT_DoubleMu3_Vtx2cm",
        "HLT_DoubleMu3_Vtx2mm",
        "HLT_DoubleMu3_JPsi",
        "HLT_DoubleMu3_Upsilon",
        "HLT_DoubleMu7_Z",
        "HLT_DoubleMu3_SameSign",
        "HLT_DoubleMu3_Psi2S"
    ),


)

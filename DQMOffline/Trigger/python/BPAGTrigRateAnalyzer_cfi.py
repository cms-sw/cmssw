import FWCore.ParameterSet.Config as cms

bpagTrigOffDQM = cms.EDAnalyzer("BPAGTrigAnalyzer",

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
	  collectionName = cms.untracked.string ("probeAnyMuon"),
	  trackCollection = cms.untracked.string ("innerTrack"),
	  requiredTriggers = cms.untracked.vstring(""),
 	  d0cut = cms.untracked.double(2.0),
 	  z0cut = cms.untracked.double(25.0), 	  
 	  recoCuts = cms.untracked.string ("pt > 0.5 && abs(eta) < 2.1 "),
 	  hltCuts  = cms.untracked.string ("pt > 0.5 && abs(eta) < 2.1 "),

	  # define the tag collection here
	  tagCollectionName = cms.untracked.string ("tagForProbeAnyMuon"),
	  tagTrackCollection = cms.untracked.string ("innerTrack"),
	  # this is the how you specify the trigger you want the probe to pass
	  # for now, the tag trigger = probe trigger
	  tagObjectTrigger = cms.untracked.string(""),
 	  tagD0cut = cms.untracked.double(2.0),
 	  tagZ0cut = cms.untracked.double(50.0), 
 	  tagRecoCuts = cms.untracked.string ("pt > 1.0 && abs(eta) < 2.1 "),
 	  tagHltCuts  = cms.untracked.string ("pt > 1.0 && abs(eta) < 2.1 "),
	  	  
 	),

	
	
									  
    ),

    # Set the ranges and numbers of bins for histograms
	# max pt is not very useful
    MaxPtParameters    = cms.untracked.vdouble(40,0.,80.),
    # PtParmeters is the low bin edges, with size = nbins + 1 
	PtParameters       = cms.untracked.vdouble(0.0, 3.0, 4.5, 6, 8, 20.0),
    EtaParameters      = cms.untracked.vdouble(4, -2.1,2.1),
    PhiParameters      = cms.untracked.vdouble(4, -3.15,3.15),
    ResParameters      = cms.untracked.vdouble(50, -0.15, 0.15),
	DrParameters       = cms.untracked.vdouble(50, 0.0, 0.05),			

    MassParameters     = cms.untracked.vdouble(50, 2.6, 3.6),

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
    #CrossSection = cms.double(0.97),
	# ddd 								
    #NSigmas90 = cms.untracked.vdouble(3.0, 3.0, 3.0, 3.0),


	# list of triggers
    # any triggers not in the hlt configuraiton
    # will be ignored
									
	TriggerNames = cms.vstring(
        "HLT_Mu3",
        "HLT_Mu5",
        "HLT_Mu9",        
    ),


)

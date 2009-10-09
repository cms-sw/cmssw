import FWCore.ParameterSet.Config as cms

quadJetAna = cms.EDAnalyzer("QuadJetAnalyzer",

    HltProcessName = cms.string("HLT"),

   # The muon collection tag must point to a
   # a group of muon-type objects, not just track objects									
   RecoMuonInputTag = cms.InputTag("muons", "", ""),
   BeamSpotInputTag = cms.InputTag("offlineBeamSpot", "", ""),
   HltRawInputTag = cms.InputTag("hltTriggerSummaryRAW", "", ""),
   HltAodInputTag = cms.InputTag("hltTriggerSummaryAOD", "", ""),
   # This is still used
   # to select based on trigger
   TriggerResultLabel = cms.InputTag("TriggerResults","","HLT"),

    # Set the ranges and numbers of bins for histograms
	# max pt is not very useful
    MaxPtParameters    = cms.vdouble(25,0.,100.),
    # PtParmeters is not currently used

    EtaParameters      = cms.vdouble(25, -3.5,3.5),
    PhiParameters      = cms.vdouble(25, -3.15,3.15),
    ResParameters      = cms.vdouble(25, -0.15, 0.15),
	DrParameters       = cms.vdouble(25, 0.0, 0.05),			

    # Use Pt Parameters to set bin edges


    # If you have cosmic matching
	# you will ignore the delta R cuts								
								   									
	L1DrCut   = cms.untracked.double(0.4),
	L2DrCut   = cms.untracked.double(0.25),
	L3DrCut   = cms.untracked.double(0.025),								
									

    DQMStore = cms.untracked.bool(True),

	# list of triggers
    # any triggers not in the hlt configuraiton
    # will be ignored
									
	TriggerNames = cms.vstring(
	"QuadJet15U"	
    ),


)

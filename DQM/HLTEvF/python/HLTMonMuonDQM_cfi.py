import FWCore.ParameterSet.Config as cms

hltMonMuDQM = cms.EDAnalyzer("HLTMuonDQMSource",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(
    	# L1 muon
	cms.PSet(
		HLTCollectionLevel = cms.string("L1"),
		HLTCollectionLabels = cms.string("HLT_L1MuOpen"),
		HLTCollectionL1seed = cms.InputTag("hltL1sL1MuOpen", "", "HLT"),
		HLTCollectionL1filter = cms.InputTag("hltL1MuOpenL1Filtered0", "", "HLT"),
		HLTCollectionL2filter = cms.InputTag("", "", "HLT"),
		HLTCollectionL2isofilter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3filter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3isofilter = cms.InputTag("", "", "HLT")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L1"),
		HLTCollectionLabels = cms.string("HLT_L1Mu"),
		HLTCollectionL1seed = cms.InputTag("hltL1sL1Mu", "", "HLT"),
		HLTCollectionL1filter = cms.InputTag("hltL1MuL1Filtered0", "", "HLT"),
		HLTCollectionL2filter = cms.InputTag("", "", "HLT"),
		HLTCollectionL2isofilter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3filter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3isofilter = cms.InputTag("", "", "HLT")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L1"),
		HLTCollectionLabels = cms.string("HLT_L1Mu20"),
		HLTCollectionL1seed = cms.InputTag("hltL1sL1SingleMu20", "", "HLT"),
		HLTCollectionL1filter = cms.InputTag("hltL1Mu20L1Filtered20", "", "HLT"),
		HLTCollectionL2filter = cms.InputTag("", "", "HLT"),
		HLTCollectionL2isofilter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3filter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3isofilter = cms.InputTag("", "", "HLT")
	),
    	# L2 muon
	cms.PSet(
		HLTCollectionLevel = cms.string("L2"),
		HLTCollectionLabels = cms.string("HLT_L2Mu9"),
		HLTCollectionL1seed = cms.InputTag("hltL1sL1SingleMu7", "", "HLT"),
		HLTCollectionL1filter = cms.InputTag("hltL1SingleMu7L1Filtered0", "", "HLT"),
		HLTCollectionL2filter = cms.InputTag("hltL2Mu9L2Filtered9", "", "HLT"),
		HLTCollectionL2isofilter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3filter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3isofilter = cms.InputTag("", "", "HLT")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L2"),
		HLTCollectionLabels = cms.string("HLT_L2Mu15"),
		HLTCollectionL1seed = cms.InputTag("hltL1sL1SingleMu7", "", "HLT"),
		HLTCollectionL1filter = cms.InputTag("hltL1SingleMu7L1Filtered0", "", "HLT"),
		HLTCollectionL2filter = cms.InputTag("hltL2Mu15L2Filtered15", "", "HLT"),
		HLTCollectionL2isofilter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3filter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3isofilter = cms.InputTag("", "", "HLT")
	),
    	# L3 muon
	cms.PSet(
		HLTCollectionLevel = cms.string("L3"),
		HLTCollectionLabels = cms.string("HLT_Mu3"),
		HLTCollectionL1seed = cms.InputTag("hltL1sL1SingleMu3", "", "HLT"),
		HLTCollectionL1filter = cms.InputTag("hltL1SingleMu3L1Filtered0", "", "HLT"),
		HLTCollectionL2filter = cms.InputTag("hltSingleMu3L2Filtered3", "", "HLT"),
		HLTCollectionL2isofilter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3filter = cms.InputTag("hltSingleMu3L3Filtered3", "", "HLT"),
		HLTCollectionL3isofilter = cms.InputTag("", "", "HLT")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L3ISO"),
		HLTCollectionLabels = cms.string("HLT_IsoMu3"),
		HLTCollectionL1seed = cms.InputTag("hltL1sL1SingleMu3", "", "HLT"),
		HLTCollectionL1filter = cms.InputTag("hltSingleMuIsoL1Filtered3", "", "HLT"),
		HLTCollectionL2filter = cms.InputTag("hltSingleMuIsoL2PreFiltered3", "", "HLT"),
		HLTCollectionL2isofilter = cms.InputTag("hltSingleMuIsoL2IsoFiltered3", "", "HLT"),
		HLTCollectionL3filter = cms.InputTag("hltSingleMuIsoL3PreFiltered3", "", "HLT"),
		HLTCollectionL3isofilter = cms.InputTag("hltSingleMuIsoL3IsoFiltered3", "", "HLT")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L3"),
		HLTCollectionLabels = cms.string("HLT_Mu9"),
		HLTCollectionL1seed = cms.InputTag("hltL1sL1SingleMu7", "", "HLT"),
		HLTCollectionL1filter = cms.InputTag("hltL1SingleMu7L1Filtered0", "", "HLT"),
		HLTCollectionL2filter = cms.InputTag("hltSingleMu9L2Filtered7", "", "HLT"),
		HLTCollectionL2isofilter = cms.InputTag("", "", "HLT"),
		HLTCollectionL3filter = cms.InputTag("hltSingleMu9L3Filtered9", "", "HLT"),
		HLTCollectionL3isofilter = cms.InputTag("", "", "HLT")
	)
    ),
    disableROOToutput = cms.untracked.bool(True)
)



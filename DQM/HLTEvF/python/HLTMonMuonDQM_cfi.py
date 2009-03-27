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
		HLTCollectionLabels = cms.string("HLT_L1MuOpen")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L1"),
		HLTCollectionLabels = cms.string("HLT_L1Mu")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L1"),
		HLTCollectionLabels = cms.string("HLT_L1Mu20")
	),
    	# L2 muon
	cms.PSet(
		HLTCollectionLevel = cms.string("L2"),
		HLTCollectionLabels = cms.string("HLT_L2Mu9")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L2"),
		HLTCollectionLabels = cms.string("HLT_L2Mu15")
	),
    	# L3 muon
	cms.PSet(
		HLTCollectionLevel = cms.string("L3"),
		HLTCollectionLabels = cms.string("HLT_Mu3")
	),
	cms.PSet(
		HLTCollectionLevel = cms.string("L3"),
		HLTCollectionLabels = cms.string("HLT_Mu9")
	)
    ),
    disableROOToutput = cms.untracked.bool(True)
)



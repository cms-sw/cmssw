import FWCore.ParameterSet.Config as cms

hltMonMuDQM = cms.EDAnalyzer("HLTMuonDQMSource",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    TrigResultInput = cms.InputTag('TriggerResults','','HLTonline'),
    filters = cms.VPSet(
    	# L1 muon
	cms.PSet(
		directoryName = cms.string('L1PassThrough'),
		triggerBits = cms.vstring('HLT_L1SingleMu10_v1','HLT_L1SingleMu20_v1','HLT_L1SingleMuOpen_DT_v1','HLT_L1SingleMuOpen_v1')
	),
    	# L2 muon
	cms.PSet(
		directoryName = cms.string('L2PassThrough'),
		triggerBits = cms.vstring('HLT_L2Mu10_v2','HLT_L2Mu20_v2')
	),
    	# L3 muon
	cms.PSet(
		directoryName = cms.string('L3Triggers'),
		triggerBits = cms.vstring('HLT_Mu20_v2','HLT_Mu24_v2','HLT_Mu30_v2','HLT_Mu3_v4','HLT_Mu5_v4','HLT_Mu8_v2','HLT_IsoMu12_v2','HLT_IsoMu15_v6','HLT_IsoMu17_v6','HLT_IsoMu24_v2','HLT_IsoMu30_v2')
	),
    	# DoubleMu
	cms.PSet(
		directoryName = cms.string('DoubleMu'),
		triggerBits = cms.vstring('HLT_L1DoubleMu0_v1','HLT_L2DoubleMu0_v2','HLT_DoubleMu3_v4','HLT_DoubleMu6_v2','HLT_DoubleMu7_v2')
	),
	# No tracker
	cms.PSet(
		directoryName = cms.string('Cosmics'),
		triggerBits = cms.vstring('HLT_L2DoubleMu35_NoVertex_v1')
	)
    	# JetStream
	#cms.PSet(
	#	directoryName = cms.string('JetStream'),
	#	triggerBits = cms.vstring('HLT_Jet15U', 'HLT_Jet30U')
	#)
    ),
    disableROOToutput = cms.untracked.bool(True)
)



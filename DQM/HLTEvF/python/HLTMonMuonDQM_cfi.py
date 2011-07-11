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
		triggerBits = cms.vstring('HLT_L1SingleMu10_v4','HLT_L1SingleMu20_v4','HLT_L1SingleMuOpen_DT_v4','HLT_L1SingleMuOpen_v4')
	),
    	# L2 muon
	cms.PSet(
		directoryName = cms.string('L2PassThrough'),
		triggerBits = cms.vstring('HLT_L2Mu10_v5','HLT_L2Mu20_v5')
	),
    	# L3 muon no iso
	cms.PSet(
		directoryName = cms.string('L3Triggers'),
		triggerBits = cms.vstring('HLT_Mu20_v5','HLT_Mu24_v5','HLT_Mu30_v5', 'HLT_Mu40_v3', 'HLT_Mu3_v7','HLT_Mu5_v7','HLT_Mu8_v5')
	),
    	# L3 muon with iso
	cms.PSet(
		directoryName = cms.string('L3Triggers_ISO'),
		triggerBits = cms.vstring('HLT_IsoMu12_v7','HLT_IsoMu15_v11','HLT_IsoMu17_v11','HLT_IsoMu24_v7','HLT_IsoMu30_v7')
	),
    	# DoubleMu
	cms.PSet(
		directoryName = cms.string('DoubleMu'),
		triggerBits = cms.vstring('HLT_L1DoubleMu0_v4','HLT_L2DoubleMu0_v6','HLT_DoubleMu3_v7','HLT_DoubleMu6_v5','HLT_DoubleMu7_v5','HLT_Mu13_Mu8_v4','HLT_Mu17_Mu8_v4')
	),
	# No tracker
	cms.PSet(
		directoryName = cms.string('Cosmics'),
		triggerBits = cms.vstring('HLT_L2DoubleMu23_NoVertex_v5', 'HLT_L1SingleMuOpen_AntiBPTX_v3')
	)
    	# JetStream
	#cms.PSet(
	#	directoryName = cms.string('JetStream'),
	#	triggerBits = cms.vstring('HLT_Jet15U', 'HLT_Jet30U')
	#)
    ),
    disableROOToutput = cms.untracked.bool(True)
)



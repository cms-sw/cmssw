import FWCore.ParameterSet.Config as cms


################# Muon HLT Quality Tests  #########################
qTesterMuonHLT = cms.EDFilter("QualityTester",
     qtList = cms.untracked.FileInPath('DQMOffline/Trigger/data/MuonHLT_QualityTests.xml'),
	 getQualityTestsFromFile = cms.untracked.bool(True),						  
     prescaleFactor = cms.untracked.int32(1),
     testInEventloop = cms.untracked.bool(False),
     verboseQT =  cms.untracked.bool(True),
     qtOnEndLumi = cms.untracked.bool(False),
	 qtOnEndRun = cms.untracked.bool(True),
	 qtestOnEndJob =   cms.untracked.bool(False),					  							  
 )

muonHLTCertSeq = cms.Sequence(qTesterMuonHLT)

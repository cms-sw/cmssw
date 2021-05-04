import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

################# Muon HLT Quality Tests  #########################
from DQMServices.Core.DQMQualityTester import DQMQualityTester
qTesterMuonHLT = DQMQualityTester(
    qtList = cms.untracked.FileInPath(
        'DQMOffline/Trigger/data/MuonHLT_QualityTests.xml'
    ),
    getQualityTestsFromFile = cms.untracked.bool(True),
    prescaleFactor = cms.untracked.int32(1),
    testInEventloop = cms.untracked.bool(False),
    verboseQT =  cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
        qtestOnEndRun = cms.untracked.bool(True),
        qtestOnEndJob =   cms.untracked.bool(True),
        #reportThreshold = cms.untracked.string("black")
)

muonHLTCertSummary = DQMEDHarvester("HLTMuonCertSummary",
    verbose = cms.untracked.bool(False),
)

muonHLTCertSeq = cms.Sequence(
    qTesterMuonHLT *
    muonHLTCertSummary
)

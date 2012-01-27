import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLoggerClient_cfi import *

dqmMessageLoggerQTester = cms.EDAnalyzer("QualityTester",
                         qtList = cms.untracked.FileInPath('DQMServices/Components/data/DQMMessageLoggerQualityTests.xml'),
                         prescaleFactor = cms.untracked.int32(1),
                         getQualityTestsFromFile = cms.untracked.bool(True)
                        )

DQMMessageLoggerClientSeq = cms.Sequence(DQMMessageLoggerClient*dqmMessageLoggerQTester)

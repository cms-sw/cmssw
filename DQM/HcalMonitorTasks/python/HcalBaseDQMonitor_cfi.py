import FWCore.ParameterSet.Config as cms

hcalBaseDQMonitor=cms.EDAnalyzer("HcalBaseDQMonitor",
                                 debug                  = cms.untracked.int32(0),
                                 online                 = cms.untracked.bool(False),
                                 AllowedCalibTypes      = cms.untracked.vint32([0,1,2,3,4,5,6,7]),
                                 mergeRuns              = cms.untracked.bool(False),
                                 enableCleanup          = cms.untracked.bool(False),
                                 subSystemFolder        = cms.untracked.string("Hcal/"),
                                 TaskFolder             = cms.untracked.string("Test/"),
                                 skipOutOfOrderLS       = cms.untracked.bool(False),
                                 NLumiBlocks            = cms.untracked.int32(4000),
                                 )

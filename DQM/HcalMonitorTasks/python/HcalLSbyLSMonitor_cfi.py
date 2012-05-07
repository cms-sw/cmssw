import FWCore.ParameterSet.Config as cms

hcalLSbyLSMonitor=cms.EDAnalyzer("HcalLSbyLSMonitor",
                                 # base class stuff
                                 debug                  = cms.untracked.int32(0),
                                 online                 = cms.untracked.bool(False),
                                 AllowedCalibTypes      = cms.untracked.vint32([0,1,2,3,4,5,6,7]),
                                 mergeRuns              = cms.untracked.bool(False),
                                 enableCleanup          = cms.untracked.bool(False),
                                 subSystemFolder        = cms.untracked.string("Hcal/"),
                                 TaskFolder             = cms.untracked.string("LSbyLS_Hcal/"),
                                 skipOutOfOrderLS       = cms.untracked.bool(True),
                                 NLumiBlocks            = cms.untracked.int32(4000),
                                 makeDiagnostics        = cms.untracked.bool(False),
                                 
                                 # List directories of all tasks that contribute to this test
                                 # Make sure that all listed tasks are filling their ProblemCurrentLB histogram,
                                 # or they will cause this test to automatically fail!
                                 TaskDirectories        = cms.untracked.vstring("DeadCellMonitor_Hcal/",
                                                                                "DigiMonitor_Hcal/",
                                                                                "HotCellMonitor_Hcal/",
                                                                                "BeamMonitor_Hcal/"),
                                 minEvents              = cms.untracked.int32(500)
                                 )

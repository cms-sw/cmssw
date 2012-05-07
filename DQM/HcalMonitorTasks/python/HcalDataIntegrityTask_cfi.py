import FWCore.ParameterSet.Config as cms

hcalDataIntegrityMonitor = cms.EDAnalyzer("HcalDataIntegrityTask",
                                          # base class stuff
                                          debug                  = cms.untracked.int32(0),
                                          online                 = cms.untracked.bool(False),
                                          AllowedCalibTypes      = cms.untracked.vint32([0,1,2,3,4,5,6,7]),
                                          mergeRuns              = cms.untracked.bool(False),
                                          enableCleanup          = cms.untracked.bool(False),
                                          subSystemFolder        = cms.untracked.string("Hcal/"),
                                          TaskFolder             = cms.untracked.string("FEDIntegrity/"),
                                          skipOutOfOrderLS       = cms.untracked.bool(False),
                                          NLumiBlocks            = cms.untracked.int32(4000),
                                          
                                          # task-specific stuff
                                   
                                          RawDataLabel           = cms.untracked.InputTag("source"),
                                          UnpackerReportLabel    = cms.untracked.InputTag("hcalDigis")
                                          )

import FWCore.ParameterSet.Config as cms

hcalRawDataMonitor=cms.EDAnalyzer("HcalRawDataMonitor",
                                 # base class stuff
                                 debug                  = cms.untracked.int32(0),
                                 online                 = cms.bool(False),
                                 AllowedCalibTypes      = cms.untracked.vint32(0), # by default, don't include calibration events
                                 mergeRuns              = cms.bool(False),
                                 enableCleanup          = cms.untracked.bool(False),
                                 subSystemFolder        = cms.untracked.string("Hcal/"),
                                 TaskFolder             = cms.untracked.string("RawDataMonitor_Hcal/"),
                                 skipOutOfOrderLS       = cms.untracked.bool(False),
                                 NLumiBlocks            = cms.untracked.int32(4000),

                                 # Collection to get
                                 FEDRawDataCollection = cms.untracked.InputTag("source"),
                                 digiLabel            = cms.untracked.InputTag("hcalDigis"),
                                  
                                 excludeHORing2         = cms.untracked.bool(True),                                  
                                 )

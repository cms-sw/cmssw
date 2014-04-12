import FWCore.ParameterSet.Config as cms

hcalDetDiagLEDMonitor=cms.EDAnalyzer("HcalDetDiagLEDMonitor",
                                     # base class stuff
                                     debug                  = cms.untracked.int32(0),
                                     online                 = cms.untracked.bool(False),
                                     AllowedCalibTypes      = cms.untracked.vint32(1,2,3,4,5),
                                     mergeRuns              = cms.untracked.bool(False),
                                     enableCleanup          = cms.untracked.bool(False),
                                     subSystemFolder        = cms.untracked.string("Hcal/"),
                                     TaskFolder             = cms.untracked.string("DetDiagLEDMonitor_Hcal/"),
                                     skipOutOfOrderLS       = cms.untracked.bool(True),
                                     NLumiBlocks            = cms.untracked.int32(4000),
                                     makeDiagnostics        = cms.untracked.bool(False),
                                     
                                     # DetDiag LED Monitor-specific Info
                                     LEDMeanThreshold       = cms.untracked.double(0.1),
                                     LEDRmsThreshold        = cms.untracked.double(0.1),
                                     LEDReferenceData       = cms.untracked.string(""),
                                     OutputFilePath         = cms.untracked.string(""),
			             XmlFilePath            = cms.untracked.string(""),
                                     digiLabel              = cms.untracked.InputTag("hcalDigis"),
                                     calibDigiLabel         = cms.untracked.InputTag("hcalDigis"),
                                     triggerLabel           = cms.untracked.InputTag("l1GtUnpack"),
                                     hcalTBTriggerDataTag   = cms.InputTag("tbunpack")
                                   )

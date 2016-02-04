import FWCore.ParameterSet.Config as cms

hcalDetDiagTimingMonitor=cms.EDAnalyzer("HcalDetDiagTimingMonitor",
                                       # base class stuff
                                       debug                  = cms.untracked.int32(0),
                                       online                 = cms.untracked.bool(False),
                                       AllowedCalibTypes      = cms.untracked.vint32(0),  # exclude calibration data from timing monitor?
                                       mergeRuns              = cms.untracked.bool(False),
                                       enableCleanup          = cms.untracked.bool(False),
                                       subSystemFolder        = cms.untracked.string("Hcal/"),
                                       TaskFolder             = cms.untracked.string("DetDiagTimingMonitor_Hcal/"),
                                       skipOutOfOrderLS       = cms.untracked.bool(True),
                                       NLumiBlocks            = cms.untracked.int32(4000),
                                       makeDiagnostics        = cms.untracked.bool(False),
                                       
                                       # DetDiag Noise Monitor-specific Info
                                       GCTTriggerBit1         = cms.untracked.int32(15),
                                       GCTTriggerBit2         = cms.untracked.int32(16),
                                       GCTTriggerBit3         = cms.untracked.int32(17),
                                       GCTTriggerBit4         = cms.untracked.int32(18),
                                       GCTTriggerBit5         = cms.untracked.int32(19),
                                       CosmicsCorr            = cms.untracked.bool(False),
                                       gtLabel                = cms.untracked.InputTag("l1GtUnpack"),
                                       digiLabel              = cms.untracked.InputTag("hcalDigis"),
                                       FEDRawDataCollection   = cms.untracked.InputTag("source")
                                       )

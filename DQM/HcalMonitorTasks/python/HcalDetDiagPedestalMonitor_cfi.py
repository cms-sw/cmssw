import FWCore.ParameterSet.Config as cms

hcalDetDiagPedestalMonitor=cms.EDAnalyzer("HcalDetDiagPedestalMonitor",
                                          # base class stuff
                                          debug                  = cms.untracked.int32(0),
                                          online                 = cms.untracked.bool(False),
                                          AllowedCalibTypes      = cms.untracked.vint32(1),  # pedestals have calibration type 1
                                          mergeRuns              = cms.untracked.bool(False),
                                          enableCleanup          = cms.untracked.bool(False),
                                          subSystemFolder        = cms.untracked.string("Hcal/"),
                                          TaskFolder             = cms.untracked.string("DetDiagPedestalMonitor_Hcal/"),
                                          skipOutOfOrderLS       = cms.untracked.bool(True),
                                          NLumiBlocks            = cms.untracked.int32(4000),
                                          makeDiagnostics        = cms.untracked.bool(False),
                                          
                                          # DetDiag Pedestal Monitor-specific Info
                                          
                                          # Input collections
                                          digiLabel              = cms.untracked.InputTag("hcalDigis"),
                                          rawDataLabel           = cms.untracked.InputTag("source"),
                                          # reference dataset path + filename
                                          PedestalReferenceData  = cms.untracked.string(""),
                                          # path to store datasets for current run
                                          OutputFilePath         = cms.untracked.string(""),
                                          # path to store HTML output
                                          BaseHTMLpath           = cms.untracked.string(""),
                                          enableHTML             = cms.untracked.bool(False),
                                          # thresholds
                                          HBMeanTreshold         = cms.untracked.double(0.2),
                                          HBRmsTreshold          = cms.untracked.double(0.3),
                                          HEMeanTreshold         = cms.untracked.double(0.2),
                                          HERmsTreshold          = cms.untracked.double(0.3),
                                          HOMeanTreshold         = cms.untracked.double(0.2),
                                          HORmsTreshold          = cms.untracked.double(0.3),
                                          HFMeanTreshold         = cms.untracked.double(0.2),
                                          HFRmsTreshold          = cms.untracked.double(0.3)
                                   )

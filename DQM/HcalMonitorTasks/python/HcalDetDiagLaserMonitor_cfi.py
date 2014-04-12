import FWCore.ParameterSet.Config as cms

hcalDetDiagLaserMonitor=cms.EDAnalyzer("HcalDetDiagLaserMonitor",
                                   # base class stuff
                                   debug                  = cms.untracked.int32(0),
                                   online                 = cms.untracked.bool(False),
                                   AllowedCalibTypes      = cms.untracked.vint32(1,2,3,4,5),
                                   mergeRuns              = cms.untracked.bool(False),
                                   enableCleanup          = cms.untracked.bool(False),
                                   subSystemFolder        = cms.untracked.string("Hcal/"),
                                   TaskFolder             = cms.untracked.string("DetDiagLaserMonitor_Hcal/"),
                                   skipOutOfOrderLS       = cms.untracked.bool(True),
                                   NLumiBlocks            = cms.untracked.int32(4000),
                                   makeDiagnostics        = cms.untracked.bool(False),
                                   
                                   # DetDiag Laser Monitor-specific Info
                                   
                                   # Input collections
                                   digiLabel              = cms.untracked.InputTag("hcalDigis"),
                                   calibDigiLabel         = cms.untracked.InputTag("hcalDigis"),
                                   RawDataLabel           = cms.untracked.InputTag("source"),
                                   hcalTBTriggerDataTag   = cms.InputTag("tbunpack"),
                                   # reference dataset path + filename
                                   LaserReferenceData     = cms.untracked.string(""),
                                   # processed dataset name (to create HTML only)
                                   LaserDatasetName       = cms.untracked.string(""),
                                   # html path (to create HTML from dataset only)
                                   htmlOutputPath         = cms.untracked.string(""),
                                   # Save output to different files on same file
                                   Overwrite              = cms.untracked.bool(True),
                                   # path to store datasets for current run
                                   OutputFilePath         = cms.untracked.string(""),
                                   # path to store xmz.zip file to be uploaded into OMDG
                                   XmlFilePath            = cms.untracked.string(""),
			           # thresholds
                                   LaserTimingThreshold   = cms.untracked.double(0.2),
                                   LaserEnergyThreshold   = cms.untracked.double(0.1)
                                   )

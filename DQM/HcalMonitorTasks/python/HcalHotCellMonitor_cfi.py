import FWCore.ParameterSet.Config as cms

hcalHotCellMonitor=cms.EDAnalyzer("HcalHotCellMonitor",
                                  # base class stuff
                                  debug                  = cms.untracked.int32(0),
                                  online                 = cms.untracked.bool(False),
                                  AllowedCalibTypes      = cms.untracked.vint32(0), # by default, don't include calibration events
                                  mergeRuns              = cms.untracked.bool(False),
                                  enableCleanup          = cms.untracked.bool(False),
                                  subSystemFolder        = cms.untracked.string("Hcal/"),
                                  TaskFolder             = cms.untracked.string("HotCellMonitor_Hcal/"),
                                  skipOutOfOrderLS       = cms.untracked.bool(True),
                                  NLumiBlocks            = cms.untracked.int32(4000),
                                  makeDiagnostics        = cms.untracked.bool(True),
                                  
                                  # Hot Cell Monitor-specific Info
                                  
                                  # Input collections
                                  hbheRechitLabel              = cms.untracked.InputTag("hbhereco"),
                                  hoRechitLabel                = cms.untracked.InputTag("horeco"),
                                  hfRechitLabel                = cms.untracked.InputTag("hfreco"),

                                  # disable testing of HO ring 2
                                  excludeHORing2  = cms.untracked.bool(False),
                                  
                                  # Booleans for various tests
                                  test_energy     = cms.untracked.bool(False),  # dropped in favor of ET test
                                  test_et         = cms.untracked.bool(True),
                                  test_persistent = cms.untracked.bool(True),
                                  test_neighbor   = cms.untracked.bool(False),

                                  # Threshold requirements
                                  minEvents       = cms.untracked.int32(200),
                                  minErrorFlag    = cms.untracked.double(0.10), # fraction of a lumi section for which a channel must be above threshold to be considered a problem in LS plots
                                  energyThreshold = cms.untracked.double(10.),
                                  energyThreshold_HF = cms.untracked.double(20.),
                                  ETThreshold        = cms.untracked.double(5.0),
                                  ETThreshold_HF     = cms.untracked.double(5.0),
                                  # other subdetector thresholds are also untracked

                                  persistentThreshold = cms.untracked.double(6.),
                                  persistentThreshold_HF = cms.untracked.double(10.),

                                  persistentETThreshold    = cms.untracked.double(3.),
                                  persistentETThreshold_HF = cms.untracked.double(3.),
                             	  FEDRawDataCollection=cms.untracked.InputTag("rawDataCollector")

                                  )

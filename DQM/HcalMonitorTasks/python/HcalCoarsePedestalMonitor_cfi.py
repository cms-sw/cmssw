import FWCore.ParameterSet.Config as cms

hcalCoarsePedestalMonitor=cms.EDAnalyzer("HcalCoarsePedestalMonitor",
                                         # base class stuff
                                         debug                  = cms.untracked.int32(0),
                                         online                 = cms.untracked.bool(False),
                                         AllowedCalibTypes      = cms.untracked.vint32(1), # by default, only include pedestal events
                                         mergeRuns              = cms.untracked.bool(False),
                                         enableCleanup          = cms.untracked.bool(False),
                                         subSystemFolder        = cms.untracked.string("Hcal/"),
                                         TaskFolder             = cms.untracked.string("CoarsePedestalMonitor_Hcal/"),
                                         skipOutOfOrderLS       = cms.untracked.bool(False),
                                         NLumiBlocks            = cms.untracked.int32(4000),
                                         makeDiagnostics        = cms.untracked.bool(False),
                                         
                                         # Coarse Pedestal Monitor Info
                                         digiLabel              = cms.untracked.InputTag("hcalDigis"),
                                         ADCDiffThresh          = cms.untracked.double(1.),  # minimum threshold for assigning error
                                         minEvents              = cms.untracked.int32(25),
                                         # Turn off calculation of Ring2 pedestals
                                         excludeHORing2         = cms.untracked.bool(True),
                             	 	 FEDRawDataCollection=cms.untracked.InputTag("rawDataCollector")
                                         )

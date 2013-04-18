import FWCore.ParameterSet.Config as cms

hcalDigiMonitor=cms.EDAnalyzer("HcalDigiMonitor",
                               # base class stuff
                               debug                  = cms.untracked.int32(0),
                               online                 = cms.untracked.bool(False),
                               AllowedCalibTypes      = cms.untracked.vint32(0), # by default, don't include calibration events
                               mergeRuns              = cms.untracked.bool(False),
                               enableCleanup          = cms.untracked.bool(False),
                               subSystemFolder        = cms.untracked.string("Hcal/"),
                               TaskFolder             = cms.untracked.string("DigiMonitor_Hcal/"),
                               skipOutOfOrderLS       = cms.untracked.bool(False),
                               NLumiBlocks            = cms.untracked.int32(4000),
                               makeDiagnostics        = cms.untracked.bool(False),
                               
                               # Digi Monitor Info
                               digiLabel              = cms.untracked.InputTag("hcalDigis"),
                               # Shape thresh are sum of ADC counts above nominal pedestal of 3*10=30
                               shapeThresh            = cms.untracked.int32(20),
                               shapeThreshHB          = cms.untracked.int32(20),
                               shapeThreshHE          = cms.untracked.int32(20),
                               shapeThreshHO          = cms.untracked.int32(20),
                               shapeThreshHF          = cms.untracked.int32(20),
                               
                               HLTResultsLabel              = cms.untracked.InputTag("TriggerResults","","HLT"),
                               # triggers required to Min Bias conditions
                               MinBiasHLTBits               = cms.untracked.vstring("HLT_MinBiasPixel_SingleTrack",
                                                                                    "HLT_L1Tech_BSC_minBias",
                                                                                    "HLT_L1Tech_BSC_minBias_OR",
                                                                                    "HLT_L1Tech_BSC_minBias_threshold1_v2",
                                                                                    "HLT_ZeroBias_v1"),
                               
                               # disable testing of HO ring 2
                               excludeHORing2  = cms.untracked.bool(True),

                               hfRechitLabel                = cms.untracked.InputTag("hfreco"),

                               # problem checks
                               checkForMissingDigis   = cms.untracked.bool(False),
                               checkCapID             = cms.untracked.bool(True),
                               checkDigiSize          = cms.untracked.bool(True),
                               checkADCsum            = cms.untracked.bool(True),
                               checkDVerr             = cms.untracked.bool(True),
                               minDigiSize            = cms.untracked.int32(10),
                               maxDigiSize            = cms.untracked.int32(10),
                               
                               # block orbit test
                               shutOffOrbitTest       = cms.untracked.bool(False),
                               ExpectedOrbitMessageTime = cms.untracked.int32(3559)
                               )

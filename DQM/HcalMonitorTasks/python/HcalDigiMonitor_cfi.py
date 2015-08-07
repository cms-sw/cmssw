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
                               FEDRawDataCollection = cms.untracked.InputTag("rawDataCollector"),

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
                                                                                    "HLT_ZeroBias"),
                               
                               # disable testing of HO ring 2
                               excludeHORing2  = cms.untracked.bool(False),
                               excludeHO1P02          = cms.untracked.bool(False),
                               excludeBadQPLL         = cms.untracked.bool(True),

                               hfRechitLabel                = cms.untracked.InputTag("hfreco"),

                               BadChannelStatusMask   = cms.untracked.int32((1<<5) | (1<<1)), # dead cells mask: up to 03.01.2001 dead cells masks keep changing... expect a final version soon.

                               # problem checks
                               checkForMissingDigis   = cms.untracked.bool(False),
                               checkCapID             = cms.untracked.bool(True),
                               checkDigiSize          = cms.untracked.bool(True),
                               checkADCsum            = cms.untracked.bool(True),
                               checkDVerr             = cms.untracked.bool(True),
                               # min/max values are inclusive, so digis are considered
                               # good if >= minDigiSize and <=badDigiSize
                               minDigiSizeHBHE        = cms.untracked.int32(10),
                               maxDigiSizeHBHE        = cms.untracked.int32(10),
                               minDigiSizeHO          = cms.untracked.int32(10),
                               maxDigiSizeHO          = cms.untracked.int32(10),
                               minDigiSizeHF          = cms.untracked.int32(4),
                               maxDigiSizeHF          = cms.untracked.int32(6),
                               
                               # block orbit test
                               shutOffOrbitTest       = cms.untracked.bool(False),
                               ExpectedOrbitMessageTime = cms.untracked.int32(3559),
                               )

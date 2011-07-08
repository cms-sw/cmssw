import FWCore.ParameterSet.Config as cms

hcalRecHitMonitor=cms.EDAnalyzer("HcalRecHitMonitor",
                                 # base class stuff
                                 debug                  = cms.untracked.int32(0),
                                 online                 = cms.untracked.bool(False),
                                 AllowedCalibTypes      = cms.untracked.vint32(0),  # by default, don't include calibration events
                                 mergeRuns              = cms.untracked.bool(False),
                                 enableCleanup          = cms.untracked.bool(False),
                                 subSystemFolder        = cms.untracked.string("Hcal/"),
                                 TaskFolder             = cms.untracked.string("RecHitMonitor_Hcal/"),
                                 skipOutOfOrderLS       = cms.untracked.bool(False),
                                 NLumiBlocks            = cms.untracked.int32(4000),
                                 makeDiagnostics        = cms.untracked.bool(False),
                                 
                                 # variables specific to HcalRecHitMonitor
                                 
                                 # Input collections
                                 hbheRechitLabel              = cms.untracked.InputTag("hbhereco"),
                                 hoRechitLabel                = cms.untracked.InputTag("horeco"),
                                 hfRechitLabel                = cms.untracked.InputTag("hfreco"),
                                 
                                 L1GTLabel                    = cms.untracked.InputTag("l1GtUnpack"),

                                 HLTResultsLabel              = cms.untracked.InputTag("TriggerResults","","HLT"),
                                 # triggers required to meet Hcal HLT or Min Bias conditions
                                 HcalHLTBits                  = cms.untracked.vstring("HLT_L1Tech_HCAL_HF_coincidence_PM",
                                                                                      "HLT_L1Tech_HCAL_HF",
                                                                                      "HLT_ActivityHF_Coincidence3",
                                                                                      "HLT_L1Tech_HCAL_HF",
                                                                                      "HLT_L1Tech_BSC_minBias_treshold1_v2"),
                                 MinBiasHLTBits               = cms.untracked.vstring("HLT_MinBiasBSC",
                                                                                      "HLT_L1Tech_BSC_minBias",
                                                                                      "HLT_MinBiasPixel_SingleTrack",
                                                                                      "HLT_L1Tech_BSC_minBias",
                                                                                      "HLT_L1Tech_BSC_minBias_OR",
                                                                                      "HLT_L1Tech_BSC_minBias_threshold1_v2",
                                                                                      "HLT_ZeroBias_v1"),
                                 
                                 # Energy thresholds for some BPTX plots
                                 energyThreshold              = cms.untracked.double(2.),
                                 ETThreshold                  = cms.untracked.double(0.),
                                 HF_energyThreshold           = cms.untracked.double(3.),
                                 HF_ETThreshold               = cms.untracked.double(0.),
                                 HO_energyThreshold           = cms.untracked.double(5.),
                                 collisiontimediffThresh      = cms.untracked.double(10.) # max time diff between HF+, HF- weighted times for some plot filling
                                 )

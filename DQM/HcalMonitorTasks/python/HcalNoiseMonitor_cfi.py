import FWCore.ParameterSet.Config as cms

hcalNoiseMonitor=cms.EDAnalyzer("HcalNoiseMonitor",
                                # base class stuff
                                debug                  = cms.untracked.int32(0),
                                online                 = cms.untracked.bool(False),
                                AllowedCalibTypes      = cms.untracked.vint32(0), # don't include calibration events, since they skew NZS ratio?
                                mergeRuns              = cms.untracked.bool(False),
                                enableCleanup          = cms.untracked.bool(False),
                                subSystemFolder        = cms.untracked.string("Hcal/"),
                                TaskFolder             = cms.untracked.string("NoiseMonitor_Hcal/"),
                                skipOutOfOrderLS       = cms.untracked.bool(False),
                                NLumiBlocks            = cms.untracked.int32(4000),
                                
                                # parameters
                                RawDataLabel           = cms.untracked.InputTag("source"),
                                
                                HLTResultsLabel        = cms.untracked.InputTag("TriggerResults","","HLT"),
                                hbheDigiLabel        = cms.untracked.InputTag("hcalDigis"),
                                hbheRechitLabel      = cms.untracked.InputTag("hbhereco"),
                                noiseLabel           = cms.untracked.InputTag("hcalnoise"),
                                nzsHLTnames            = cms.untracked.vstring('HLT_HcalPhiSym',
                                                                               'HLT_HcalNZS_8E29'),
                                NZSeventPeriod         = cms.untracked.int32(4096),
                                
                                E2E10MinEnergy         = cms.untracked.double(50),
                                MinE2E10               = cms.untracked.double(0.7),
                                MaxE2E10               = cms.untracked.double(0.96),
                                MaxHPDHitCount         = cms.untracked.int32(17),
                                MaxHPDNoOtherHitCount  = cms.untracked.int32(10),
                                MaxADCZeros            = cms.untracked.int32(10),
                                TotalZeroMinEnergy     = cms.untracked.double(10)
                                )

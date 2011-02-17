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

                              # NZS-specific parameters
                              RawDataLabel           = cms.untracked.InputTag("source"),

                              HLTResultsLabel        = cms.untracked.InputTag("TriggerResults","","HLT"),
                              nzsHLTnames            = cms.untracked.vstring('HLT_HcalPhiSym',
                                                                   'HLT_HcalNZS_8E29'),
                              NZSeventPeriod         = cms.untracked.int32(4096),
                              )

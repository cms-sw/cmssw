import FWCore.ParameterSet.Config as cms

hcalDeadCellMonitor=cms.EDAnalyzer("HcalDeadCellMonitor",
                                   # base class stuff
                                   debug                  = cms.untracked.int32(0),
                                   online                 = cms.untracked.bool(False),
                                   AllowedCalibTypes      = cms.untracked.vint32(),
                                   mergeRuns              = cms.untracked.bool(False),
                                   enableCleanup          = cms.untracked.bool(False),
                                   subSystemFolder        = cms.untracked.string("Hcal/"),
                                   TaskFolder             = cms.untracked.string("DeadCellMonitor_Hcal/"),
                                   skipOutOfOrderLS       = cms.untracked.bool(True),
                                   NLumiBlocks            = cms.untracked.int32(4000),
                                   makeDiagnostics        = cms.untracked.bool(False),
                                   BadChannelStatusMask   = cms.untracked.int32(32770), # 32770 = 0x2+0x8000 = new dead cells mask, masked at rechit and trigger
                                   # Dead Cell Monitor-specific Info
                                   
                                   # Input collections
                                   hbheRechitLabel        = cms.untracked.InputTag("hbhereco"),
                                   hoRechitLabel          = cms.untracked.InputTag("horeco"),
                                   hfRechitLabel          = cms.untracked.InputTag("hfreco"),
                                   digiLabel              = cms.untracked.InputTag("hcalDigis"),
                                   # minimum number of events necessary for lumi-block-based checking to commence
                                   minDeadEventCount      = cms.untracked.int32(1000),

                                   excludeHORing2         = cms.untracked.bool(False),
                                   #booleans for dead cell tests
                                   test_digis             = cms.untracked.bool(True), # test for recent missing digis
                                   test_rechits           = cms.untracked.bool(True), # test for missing rechits
                                   MissingRechitEnergyThreshold = cms.untracked.double(-99.)

                                   )

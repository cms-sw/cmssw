import FWCore.ParameterSet.Config as cms

hcalTrigPrimMonitor=cms.EDAnalyzer("HcalTrigPrimMonitor",
   # base class stuff
   debug                  = cms.untracked.int32(0),
   online                 = cms.untracked.bool(False),
   AllowedCalibTypes      = cms.untracked.vint32(0), # by default, don't include calibration events
   #AllowedCalibTypes      = cms.untracked.vint32(0,1,2,3,4,5),
   mergeRuns              = cms.untracked.bool(False),
   enableCleanup          = cms.untracked.bool(False),
   subSystemFolder        = cms.untracked.string("Hcal/"),
   TaskFolder             = cms.untracked.string("TrigPrimMonitor_Hcal/"),
   skipOutOfOrderLS       = cms.untracked.bool(False),
   NLumiBlocks            = cms.untracked.int32(4000),

   # TrigPrimMonitor
   dataLabel              = cms.InputTag('hcalDigis'),
   emulLabel              = cms.InputTag('valHcalTriggerPrimitiveDigis'),
   ZSBadTPThreshold       = cms.vint32( 0,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2
                                      ),                                        
   ZSAlarmThreshold       = cms.vint32( 0,
#                                       10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
#                                       10, 10, 10, 10, 40, 40, 10, 20, 20, 20,
#                                       13, 13, 13, 13, 13, 13, 10, 10,
#                                       3, 3, 2, 2
                                        10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                        10, 10, 10, 10, 40, 50, 10, 30, 30, 30,
                                        20, 15, 15, 15, 15, 15, 10, 10,
                                        5, 5, 5, 5
                                      ),
   FEDRawDataCollection=cms.untracked.InputTag("rawDataCollector")
)

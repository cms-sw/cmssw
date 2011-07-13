import FWCore.ParameterSet.Config as cms
#-----------------------------------
#  Detector State Filter
#-----------------------------------
detectorStateFilter = cms.EDFilter("DetectorStateFilter",
           DetectorType   = cms.untracked.string('sistrip'),
           DebugOn        = cms.untracked.bool(False),
           DcsStatusLabel = cms.untracked.InputTag('scalersRawToDigi')
)    
#-----------------------------------
#  Simple Event Filter
#-----------------------------------
simpleEventFilter = cms.EDFilter("SimpleEventFilter",
           EventsToSkip = cms.untracked.int32(10),
           DebugOn      = cms.untracked.bool(True)                                               
                                      
)    

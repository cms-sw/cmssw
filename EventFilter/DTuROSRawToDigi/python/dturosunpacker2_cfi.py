import FWCore.ParameterSet.Config as cms

dturosunpacker2 = cms.EDProducer("DTuROSRawToDigi2",
                                   DTuROS_FED_Source = cms.InputTag("source"),
                                   feds     = cms.untracked.vint32( 1368,),
                                   debug = cms.untracked.bool(True),
                                )

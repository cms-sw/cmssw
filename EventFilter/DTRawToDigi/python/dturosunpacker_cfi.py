import FWCore.ParameterSet.Config as cms

dturosunpacker = cms.EDProducer("DTuROSRawToDigi",
                                  inputLabel = cms.InputTag("source"),
                                  feds     = cms.untracked.vint32( 1369, 1370, 1371,),
                                  debug = cms.untracked.bool(False),
                               )

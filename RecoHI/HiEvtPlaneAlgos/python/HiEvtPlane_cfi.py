import FWCore.ParameterSet.Config as cms

hiEvtPlane = cms.EDProducer("EvtPlaneProducer",
                            useECAL_ = cms.untracked.bool(True),
                            useHCAL_ = cms.untracked.bool(True)
                            )
                            




    

import FWCore.ParameterSet.Config as cms

hiEvtPlaneFlat = cms.EDProducer('HiEvtPlaneFlatProducer',
                                vtxCollection_=cms.InputTag("hiSelectedVertex"),
                                inputPlanes_ = cms.InputTag("hiEvtPlane","recoLevel"),
                                centrality_  = cms.InputTag("centralityBin")
                                )

import FWCore.ParameterSet.Config as cms

hiEvtPlaneFlatCalib = cms.EDAnalyzer('HiEvtPlaneFlatCalib',
                                     vtxCollection_=cms.InputTag("hiSelectedVertex"),
                                     inputPlanes_ = cms.InputTag("hiEvtPlane","recoLevel"),
                                     centrality_  = cms.InputTag("centralityBin"),
                                     genFlatPsi_ = cms.untracked.bool(True)
                                     )

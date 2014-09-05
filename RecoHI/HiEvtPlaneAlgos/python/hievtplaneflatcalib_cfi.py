import FWCore.ParameterSet.Config as cms

hiEvtPlaneFlatCalib = cms.EDAnalyzer('HiEvtPlaneFlatCalib',
                                     vtxCollection_=cms.InputTag("hiSelectedVertex"),
                                     genFlatPsi_ = cms.untracked.bool(True)
                                     )

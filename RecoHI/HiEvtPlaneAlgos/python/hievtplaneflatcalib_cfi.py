import FWCore.ParameterSet.Config as cms

hiEvtPlaneFlatCalib = cms.EDProducer('HiEvtPlaneFlatCalib',
                                genFlatPsi_ = cms.untracked.bool(True),
                                genSubEvt_  = cms.untracked.bool(False)
)

import FWCore.ParameterSet.Config as cms

hiEvtPlaneFlat = cms.EDProducer('HiEvtPlaneFlatProducer',
                                genFlatPsi_ = cms.untracked.bool(True),
                                genSubEvt_  = cms.untracked.bool(False)
)

import FWCore.ParameterSet.Config as cms

L1MuTriggerPtScale = cms.ESProducer("L1MuTriggerPtScaleProducer",
    nbitPackingPt = cms.int32(5),
    scalePt = cms.vdouble(-1.0, 0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0, 1000000.0),
    signedPackingPt = cms.bool(False),
    nbinsPt = cms.int32(32)
)




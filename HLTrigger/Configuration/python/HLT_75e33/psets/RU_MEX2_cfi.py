import FWCore.ParameterSet.Config as cms

RU_MEX2 = cms.PSet(
    chi2Max = cms.double(100.0),
    chi2Norm_2D_ = cms.double(35),
    chi2_str = cms.double(50.0),
    dPhiIntMax = cms.double(0.004),
    dPhiMax = cms.double(0.006),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5),
    doCollisions = cms.bool(True),
    enlarge = cms.bool(False),
    minLayersApart = cms.int32(1),
    wideSeg = cms.double(3.0)
)
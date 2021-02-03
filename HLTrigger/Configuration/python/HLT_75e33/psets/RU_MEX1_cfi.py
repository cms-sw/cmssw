import FWCore.ParameterSet.Config as cms

RU_MEX1 = cms.PSet(
    chi2Max = cms.double(180.0),
    chi2Norm_2D_ = cms.double(60),
    chi2_str = cms.double(80.0),
    dPhiIntMax = cms.double(0.005),
    dPhiMax = cms.double(0.007),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5),
    doCollisions = cms.bool(True),
    enlarge = cms.bool(False),
    minLayersApart = cms.int32(1),
    wideSeg = cms.double(3.0)
)
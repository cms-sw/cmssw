import FWCore.ParameterSet.Config as cms

RU_ME13 = cms.PSet(
    chi2Max = cms.double(60.0),
    chi2Norm_2D_ = cms.double(20),
    chi2_str = cms.double(30.0),
    dPhiIntMax = cms.double(0.002),
    dPhiMax = cms.double(0.003),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5),
    doCollisions = cms.bool(True),
    enlarge = cms.bool(False),
    minLayersApart = cms.int32(1),
    wideSeg = cms.double(3.0)
)
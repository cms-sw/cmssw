import FWCore.ParameterSet.Config as cms

RU_ME0 = cms.PSet(
    doCollisions = cms.bool(True),
    chi2Norm_2D_ = cms.double(100),
    chi2_str = cms.double(100.0),
    chi2Max = cms.double(100.0),
    dPhiIntMax = cms.double(0.1),
    dPhiMax = cms.double(0.1),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(1),
    dRIntMax = cms.double(10.0),
    dRMax = cms.double(10.0)
)

ME0SegAlgoRU = cms.PSet(
    algo_name = cms.string('ME0SegAlgoRU'),
    algo_pset = cms.PSet(cms.PSet(RU_ME0))
)


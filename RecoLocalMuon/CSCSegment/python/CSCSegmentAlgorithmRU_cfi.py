import FWCore.ParameterSet.Config as cms

RU_ME1A = cms.PSet(
    doCollisions = cms.bool(True),
    chi2Norm_2D_ = cms.double(35),
    chi2_str_ = cms.double(50.0),
    chi2Max = cms.double(100.0),
    dPhiIntMax = cms.double(0.005),
    dPhiMax = cms.double(0.006),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(1),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5)
)
RU_ME1B = cms.PSet(
    doCollisions = cms.bool(True),
    chi2Norm_2D_ = cms.double(35),
    chi2_str_ = cms.double(50.0),
    chi2Max = cms.double(100.0),
    dPhiIntMax = cms.double(0.004),
    dPhiMax = cms.double(0.005),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(1),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5)
)
RU_ME12 = cms.PSet(
    doCollisions = cms.bool(True),
    chi2Norm_2D_ = cms.double(35),
    chi2_str_ = cms.double(50.0),
    chi2Max = cms.double(100.0),
    dPhiIntMax = cms.double(0.003),
    dPhiMax = cms.double(0.004),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(1),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5)
)
RU_ME13 = cms.PSet(
    doCollisions = cms.bool(True),
    chi2Norm_2D_ = cms.double(20),
    chi2_str_ = cms.double(30.0),
    chi2Max = cms.double(60.0),
    dPhiIntMax = cms.double(0.002),
    dPhiMax = cms.double(0.003),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(1),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5)
)
RU_MEX1 = cms.PSet(
    doCollisions = cms.bool(True),
    chi2Norm_2D_ = cms.double(60),
    chi2_str_ = cms.double(80.0),
    chi2Max = cms.double(180.0),
    dPhiIntMax = cms.double(0.005),
    dPhiMax = cms.double(0.007),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(1),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5)
)
RU_MEX2 = cms.PSet(
    doCollisions = cms.bool(True),
    chi2Norm_2D_ = cms.double(35),
    chi2_str_ = cms.double(50.0),
    chi2Max = cms.double(100.0),
    dPhiIntMax = cms.double(0.004),
    dPhiMax = cms.double(0.006),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(1),
    dRIntMax = cms.double(2.0),
    dRMax = cms.double(1.5)
)

CSCSegAlgoRU = cms.PSet(
    chamber_types = cms.vstring('ME1/a', 
        'ME1/b', 
        'ME1/2', 
        'ME1/3', 
        'ME2/1', 
        'ME2/2', 
        'ME3/1', 
        'ME3/2', 
        'ME4/1',
        'ME4/2'),
    algo_name = cms.string('CSCSegAlgoRU'),
    algo_psets = cms.VPSet(
cms.PSet(
        RU_ME1A
    ), 
cms.PSet(
        RU_ME1B
    ),
cms.PSet(
        RU_ME12
    ),
cms.PSet(
        RU_ME13
    ),
cms.PSet(
        RU_MEX1
    ),
cms.PSet(
        RU_MEX2
    )),
    parameters_per_chamber_type = cms.vint32(1, 2, 3, 4, 5, 
        6, 5, 6, 5, 6)
)


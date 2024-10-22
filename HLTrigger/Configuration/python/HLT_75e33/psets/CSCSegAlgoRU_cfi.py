import FWCore.ParameterSet.Config as cms

CSCSegAlgoRU = cms.PSet(
    algo_name = cms.string('CSCSegAlgoRU'),
    algo_psets = cms.VPSet(
        cms.PSet(
            chi2Max = cms.double(100.0),
            chi2Norm_2D_ = cms.double(35),
            chi2_str = cms.double(50.0),
            dPhiIntMax = cms.double(0.005),
            dPhiMax = cms.double(0.006),
            dRIntMax = cms.double(2.0),
            dRMax = cms.double(1.5),
            doCollisions = cms.bool(True),
            enlarge = cms.bool(False),
            minLayersApart = cms.int32(1),
            wideSeg = cms.double(3.0)
        ),
        cms.PSet(
            chi2Max = cms.double(100.0),
            chi2Norm_2D_ = cms.double(35),
            chi2_str = cms.double(50.0),
            dPhiIntMax = cms.double(0.004),
            dPhiMax = cms.double(0.005),
            dRIntMax = cms.double(2.0),
            dRMax = cms.double(1.5),
            doCollisions = cms.bool(True),
            enlarge = cms.bool(False),
            minLayersApart = cms.int32(1),
            wideSeg = cms.double(3.0)
        ),
        cms.PSet(
            chi2Max = cms.double(100.0),
            chi2Norm_2D_ = cms.double(35),
            chi2_str = cms.double(50.0),
            dPhiIntMax = cms.double(0.003),
            dPhiMax = cms.double(0.004),
            dRIntMax = cms.double(2.0),
            dRMax = cms.double(1.5),
            doCollisions = cms.bool(True),
            enlarge = cms.bool(False),
            minLayersApart = cms.int32(1),
            wideSeg = cms.double(3.0)
        ),
        cms.PSet(
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
        ),
        cms.PSet(
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
        ),
        cms.PSet(
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
    ),
    chamber_types = cms.vstring(
        'ME1/a',
        'ME1/b',
        'ME1/2',
        'ME1/3',
        'ME2/1',
        'ME2/2',
        'ME3/1',
        'ME3/2',
        'ME4/1',
        'ME4/2'
    ),
    parameters_per_chamber_type = cms.vint32(
        1, 2, 3, 4, 5,
        6, 5, 6, 5, 6
    )
)
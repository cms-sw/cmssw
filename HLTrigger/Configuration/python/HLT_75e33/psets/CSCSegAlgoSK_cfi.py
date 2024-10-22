import FWCore.ParameterSet.Config as cms

CSCSegAlgoSK = cms.PSet(
    algo_name = cms.string('CSCSegAlgoSK'),
    algo_psets = cms.VPSet(
        cms.PSet(
            chi2Max = cms.double(99999.0),
            dPhiFineMax = cms.double(0.025),
            dPhiMax = cms.double(0.003),
            dRPhiFineMax = cms.double(8.0),
            dRPhiMax = cms.double(8.0),
            minLayersApart = cms.int32(2),
            verboseInfo = cms.untracked.bool(True),
            wideSeg = cms.double(3.0)
        ),
        cms.PSet(
            chi2Max = cms.double(99999.0),
            dPhiFineMax = cms.double(0.025),
            dPhiMax = cms.double(0.025),
            dRPhiFineMax = cms.double(3.0),
            dRPhiMax = cms.double(8.0),
            minLayersApart = cms.int32(2),
            verboseInfo = cms.untracked.bool(True),
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
        2, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    )
)
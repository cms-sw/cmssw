import FWCore.ParameterSet.Config as cms

CSCSegAlgoTC = cms.PSet(
    algo_name = cms.string('CSCSegAlgoTC'),
    algo_psets = cms.VPSet(
        cms.PSet(
            SegmentSorting = cms.int32(1),
            chi2Max = cms.double(6000.0),
            chi2ndfProbMin = cms.double(0.0001),
            dPhiFineMax = cms.double(0.02),
            dPhiMax = cms.double(0.003),
            dRPhiFineMax = cms.double(6.0),
            dRPhiMax = cms.double(1.2),
            minLayersApart = cms.int32(2),
            verboseInfo = cms.untracked.bool(True)
        ),
        cms.PSet(
            SegmentSorting = cms.int32(1),
            chi2Max = cms.double(6000.0),
            chi2ndfProbMin = cms.double(0.0001),
            dPhiFineMax = cms.double(0.013),
            dPhiMax = cms.double(0.00198),
            dRPhiFineMax = cms.double(3.0),
            dRPhiMax = cms.double(0.6),
            minLayersApart = cms.int32(2),
            verboseInfo = cms.untracked.bool(True)
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
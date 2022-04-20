import FWCore.ParameterSet.Config as cms

CSCSegAlgoDF = cms.PSet(
    algo_name = cms.string('CSCSegAlgoDF'),
    algo_psets = cms.VPSet(
        cms.PSet(
            CSCSegmentDebug = cms.untracked.bool(False),
            Pruning = cms.untracked.bool(False),
            chi2Max = cms.double(5000.0),
            dPhiFineMax = cms.double(0.025),
            dRPhiFineMax = cms.double(8.0),
            dXclusBoxMax = cms.double(8.0),
            dYclusBoxMax = cms.double(8.0),
            maxDPhi = cms.double(999.0),
            maxDTheta = cms.double(999.0),
            maxRatioResidualPrune = cms.double(3.0),
            minHitsForPreClustering = cms.int32(10),
            minHitsPerSegment = cms.int32(3),
            minLayersApart = cms.int32(2),
            nHitsPerClusterIsShower = cms.int32(20),
            preClustering = cms.untracked.bool(False),
            tanPhiMax = cms.double(0.5),
            tanThetaMax = cms.double(1.2)
        ),
        cms.PSet(
            CSCSegmentDebug = cms.untracked.bool(False),
            Pruning = cms.untracked.bool(False),
            chi2Max = cms.double(5000.0),
            dPhiFineMax = cms.double(0.025),
            dRPhiFineMax = cms.double(12.0),
            dXclusBoxMax = cms.double(8.0),
            dYclusBoxMax = cms.double(12.0),
            maxDPhi = cms.double(999.0),
            maxDTheta = cms.double(999.0),
            maxRatioResidualPrune = cms.double(3.0),
            minHitsForPreClustering = cms.int32(10),
            minHitsPerSegment = cms.int32(3),
            minLayersApart = cms.int32(2),
            nHitsPerClusterIsShower = cms.int32(20),
            preClustering = cms.untracked.bool(False),
            tanPhiMax = cms.double(0.8),
            tanThetaMax = cms.double(2.0)
        ),
        cms.PSet(
            CSCSegmentDebug = cms.untracked.bool(False),
            Pruning = cms.untracked.bool(False),
            chi2Max = cms.double(5000.0),
            dPhiFineMax = cms.double(0.025),
            dRPhiFineMax = cms.double(8.0),
            dXclusBoxMax = cms.double(8.0),
            dYclusBoxMax = cms.double(8.0),
            maxDPhi = cms.double(999.0),
            maxDTheta = cms.double(999.0),
            maxRatioResidualPrune = cms.double(3.0),
            minHitsForPreClustering = cms.int32(30),
            minHitsPerSegment = cms.int32(3),
            minLayersApart = cms.int32(2),
            nHitsPerClusterIsShower = cms.int32(20),
            preClustering = cms.untracked.bool(False),
            tanPhiMax = cms.double(0.5),
            tanThetaMax = cms.double(1.2)
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
        3, 1, 2, 2, 1,
        2, 1, 2, 1, 2
    )
)

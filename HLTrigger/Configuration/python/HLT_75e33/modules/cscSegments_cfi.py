import FWCore.ParameterSet.Config as cms

cscSegments = cms.EDProducer("CSCSegmentProducer",
    algo_psets = cms.VPSet(
        cms.PSet(
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
        ),
        cms.PSet(
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
        ),
        cms.PSet(
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
        ),
        cms.PSet(
            algo_name = cms.string('CSCSegAlgoST'),
            algo_psets = cms.VPSet(
                cms.PSet(
                    BPMinImprovement = cms.double(10000.0),
                    BrutePruning = cms.bool(True),
                    CSCDebug = cms.untracked.bool(False),
                    CorrectTheErrors = cms.bool(True),
                    Covariance = cms.double(0.0),
                    ForceCovariance = cms.bool(False),
                    ForceCovarianceAll = cms.bool(False),
                    NormChi2Cut2D = cms.double(20.0),
                    NormChi2Cut3D = cms.double(10.0),
                    Pruning = cms.bool(True),
                    SeedBig = cms.double(0.0015),
                    SeedSmall = cms.double(0.0002),
                    curvePenalty = cms.double(2.0),
                    curvePenaltyThreshold = cms.double(0.85),
                    dPhiFineMax = cms.double(0.025),
                    dRPhiFineMax = cms.double(8.0),
                    dXclusBoxMax = cms.double(4.0),
                    dYclusBoxMax = cms.double(8.0),
                    hitDropLimit4Hits = cms.double(0.6),
                    hitDropLimit5Hits = cms.double(0.8),
                    hitDropLimit6Hits = cms.double(0.3333),
                    maxDPhi = cms.double(999.0),
                    maxDTheta = cms.double(999.0),
                    maxRatioResidualPrune = cms.double(3),
                    maxRecHitsInCluster = cms.int32(20),
                    minHitsPerSegment = cms.int32(3),
                    onlyBestSegment = cms.bool(False),
                    preClustering = cms.bool(True),
                    preClusteringUseChaining = cms.bool(True),
                    prePrun = cms.bool(True),
                    prePrunLimit = cms.double(3.17),
                    tanPhiMax = cms.double(0.5),
                    tanThetaMax = cms.double(1.2),
                    useShowering = cms.bool(False),
                    yweightPenalty = cms.double(1.5),
                    yweightPenaltyThreshold = cms.double(1.0)
                ),
                cms.PSet(
                    BPMinImprovement = cms.double(10000.0),
                    BrutePruning = cms.bool(True),
                    CSCDebug = cms.untracked.bool(False),
                    CorrectTheErrors = cms.bool(True),
                    Covariance = cms.double(0.0),
                    ForceCovariance = cms.bool(False),
                    ForceCovarianceAll = cms.bool(False),
                    NormChi2Cut2D = cms.double(20.0),
                    NormChi2Cut3D = cms.double(10.0),
                    Pruning = cms.bool(True),
                    SeedBig = cms.double(0.0015),
                    SeedSmall = cms.double(0.0002),
                    curvePenalty = cms.double(2.0),
                    curvePenaltyThreshold = cms.double(0.85),
                    dPhiFineMax = cms.double(0.025),
                    dRPhiFineMax = cms.double(8.0),
                    dXclusBoxMax = cms.double(4.0),
                    dYclusBoxMax = cms.double(8.0),
                    hitDropLimit4Hits = cms.double(0.6),
                    hitDropLimit5Hits = cms.double(0.8),
                    hitDropLimit6Hits = cms.double(0.3333),
                    maxDPhi = cms.double(999.0),
                    maxDTheta = cms.double(999.0),
                    maxRatioResidualPrune = cms.double(3),
                    maxRecHitsInCluster = cms.int32(24),
                    minHitsPerSegment = cms.int32(3),
                    onlyBestSegment = cms.bool(False),
                    preClustering = cms.bool(True),
                    preClusteringUseChaining = cms.bool(True),
                    prePrun = cms.bool(True),
                    prePrunLimit = cms.double(3.17),
                    tanPhiMax = cms.double(0.5),
                    tanThetaMax = cms.double(1.2),
                    useShowering = cms.bool(False),
                    yweightPenalty = cms.double(1.5),
                    yweightPenaltyThreshold = cms.double(1.0)
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
        ),
        cms.PSet(
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
    ),
    algo_type = cms.int32(5),
    inputObjects = cms.InputTag("csc2DRecHits")
)

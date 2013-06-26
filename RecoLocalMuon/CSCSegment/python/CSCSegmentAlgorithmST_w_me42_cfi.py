import FWCore.ParameterSet.Config as cms

# The following algorithms looks how far a rechit is from the
# proto segment in terms of its error ellipse.  This is different
# from the other algorithms which use a cylinder around the proto
# segment and look for rechits within that cylinder
ST_ME1234 = cms.PSet(

    #Parameters for showering segments
    useShowering = cms.bool(False),
    maxRatioResidualPrune = cms.double(3),
    dRPhiFineMax = cms.double(8.0),
    dPhiFineMax = cms.double(0.025),
    tanThetaMax = cms.double(1.2),
    tanPhiMax = cms.double(0.5),
    maxDPhi = cms.double(999.),
    maxDTheta = cms.double(999.),


    curvePenaltyThreshold = cms.double(0.85),
    minHitsPerSegment = cms.int32(3),
    yweightPenaltyThreshold = cms.double(1.0),
    curvePenalty = cms.double(2.0),
    dXclusBoxMax = cms.double(4.0),
    BrutePruning = cms.bool(True),
    BPMinImprovement = cms.double(10000.),
    yweightPenalty = cms.double(1.5),
    hitDropLimit5Hits = cms.double(0.8),
    preClustering = cms.bool(True),
    preClusteringUseChaining = cms.bool(True),
    hitDropLimit4Hits = cms.double(0.6),
    hitDropLimit6Hits = cms.double(0.3333),
    maxRecHitsInCluster = cms.int32(20),
    CSCDebug = cms.untracked.bool(False),
    onlyBestSegment = cms.bool(False),
    Pruning = cms.bool(True),
    dYclusBoxMax = cms.double(8.0),
    # Correction to improove fit
    CorrectTheErrors = cms.bool(True),
    NormChi2Cut2D = cms.double(20.0),
    NormChi2Cut3D = cms.double(10.0),
    prePrun = cms.bool(True),
    prePrunLimit = cms.double(3.17),
    SeedSmall = cms.double(0.000200),
    SeedBig = cms.double(0.001500),
    ForceCovariance = cms.bool(False),
    ForceCovarianceAll = cms.bool(False),
    Covariance = cms.double(0.0)
)
ST_ME1A = cms.PSet(

    #Parameters for showering segments
    useShowering = cms.bool(False),
    maxRatioResidualPrune = cms.double(3),
    dRPhiFineMax = cms.double(8.0),
    dPhiFineMax = cms.double(0.025),
    tanThetaMax = cms.double(1.2),
    tanPhiMax = cms.double(0.5),
    maxDPhi = cms.double(999.),
    maxDTheta = cms.double(999.),


    curvePenaltyThreshold = cms.double(0.85),
    minHitsPerSegment = cms.int32(3),
    yweightPenaltyThreshold = cms.double(1.0),
    curvePenalty = cms.double(2.0),
    dXclusBoxMax = cms.double(4.0),
    BrutePruning = cms.bool(True),
    BPMinImprovement = cms.double(10000.),
    yweightPenalty = cms.double(1.5),
    hitDropLimit5Hits = cms.double(0.8),
    preClustering = cms.bool(True),
    preClusteringUseChaining = cms.bool(True),
    hitDropLimit4Hits = cms.double(0.6),
    hitDropLimit6Hits = cms.double(0.3333),
    maxRecHitsInCluster = cms.int32(24),
    CSCDebug = cms.untracked.bool(False),
    onlyBestSegment = cms.bool(False),
    Pruning = cms.bool(True),
    dYclusBoxMax = cms.double(8.0),
    # Correction to improove fit
    CorrectTheErrors = cms.bool(True),
    NormChi2Cut2D = cms.double(20.0), 
    NormChi2Cut3D = cms.double(10.0), 
    prePrun = cms.bool(True),
    prePrunLimit = cms.double(3.17),
    SeedSmall = cms.double(0.000200),
    SeedBig = cms.double(0.001500),
    ForceCovariance = cms.bool(False),
    ForceCovarianceAll = cms.bool(False),
    Covariance = cms.double(0.0)

)
CSCSegAlgoST = cms.PSet(
    algo_name = cms.string('CSCSegAlgoST'),
    algo_psets = cms.VPSet( cms.PSet(ST_ME1234), cms.PSet(ST_ME1A) ),
    chamber_types = cms.vstring('ME1/a','ME1/b','ME1/2','ME1/3','ME2/1','ME2/2','ME3/1','ME3/2','ME4/1','ME4/2'),
    parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 1, 1, 1, 1, 1)
)


import FWCore.ParameterSet.Config as cms

# The following DF algorithms looks how far a rechit is from the
# proto segment in terms of its error ellipse.  This is different
# from SK & TC algorithms which use a cylinder around the proto
# segment and look for rechits within that cylinder
DF_ME1234_1 = cms.PSet(
    preClustering = cms.untracked.bool(False),
    minHitsPerSegment = cms.int32(3),
    dPhiFineMax = cms.double(0.025),
    chi2Max = cms.double(5000.0),
    dXclusBoxMax = cms.double(8.0),
    tanThetaMax = cms.double(1.2),
    tanPhiMax = cms.double(0.5),
    maxRatioResidualPrune = cms.double(3.0),
    minHitsForPreClustering = cms.int32(10),
    CSCSegmentDebug = cms.untracked.bool(False),
    dRPhiFineMax = cms.double(8.0),
    nHitsPerClusterIsShower = cms.int32(20),
    minLayersApart = cms.int32(2),
    Pruning = cms.untracked.bool(False),
    dYclusBoxMax = cms.double(8.0),
    maxDPhi = cms.double(999.),
    maxDTheta = cms.double(999.)
)
DF_ME1234_2 = cms.PSet(
    preClustering = cms.untracked.bool(False),
    minHitsPerSegment = cms.int32(3),
    dPhiFineMax = cms.double(0.025),
    chi2Max = cms.double(5000.0),
    dXclusBoxMax = cms.double(8.0),
    tanThetaMax = cms.double(2.0),
    tanPhiMax = cms.double(0.8),
    maxRatioResidualPrune = cms.double(3.0),
    minHitsForPreClustering = cms.int32(10),
    CSCSegmentDebug = cms.untracked.bool(False),
    dRPhiFineMax = cms.double(12.0),
    nHitsPerClusterIsShower = cms.int32(20),
    minLayersApart = cms.int32(2),
    Pruning = cms.untracked.bool(False),
    dYclusBoxMax = cms.double(12.0),
    maxDPhi = cms.double(999.),
    maxDTheta = cms.double(999.)
)
DF_ME1A = cms.PSet(
    preClustering = cms.untracked.bool(False),
    minHitsPerSegment = cms.int32(3),
    dPhiFineMax = cms.double(0.025),
    chi2Max = cms.double(5000.0),
    dXclusBoxMax = cms.double(8.0),
    tanThetaMax = cms.double(1.2),
    tanPhiMax = cms.double(0.5),
    maxRatioResidualPrune = cms.double(3.0),
    minHitsForPreClustering = cms.int32(30),
    CSCSegmentDebug = cms.untracked.bool(False),
    dRPhiFineMax = cms.double(8.0),
    nHitsPerClusterIsShower = cms.int32(20),
    minLayersApart = cms.int32(2),
    Pruning = cms.untracked.bool(False),
    dYclusBoxMax = cms.double(8.0),
    maxDPhi = cms.double(999.),
    maxDTheta = cms.double(999.)
)
CSCSegAlgoDF = cms.PSet(
    chamber_types = cms.vstring('ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1','ME4/2'),
    algo_name = cms.string('CSCSegAlgoDF'),
    algo_psets = cms.VPSet( cms.PSet(DF_ME1234_1), cms.PSet(DF_ME1234_2), cms.PSet(DF_ME1A) ),
    parameters_per_chamber_type = cms.vint32( 3, 1, 2, 2, 1, 2, 1, 2, 1, 2 )
)


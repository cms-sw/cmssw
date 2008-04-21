import FWCore.ParameterSet.Config as cms

# The following algorithms looks how far a rechit is from the
# proto segment in terms of its error ellipse.  This is different
# from the other algorithms which use a cylinder around the proto
# segment and look for rechits within that cylinder
ST_ME1234 = cms.PSet(
    curvePenaltyThreshold = cms.untracked.double(0.85),
    minHitsPerSegment = cms.untracked.int32(3),
    yweightPenaltyThreshold = cms.untracked.double(1.0),
    curvePenalty = cms.untracked.double(2.0),
    dXclusBoxMax = cms.untracked.double(4.0),
    BrutePruning = cms.untracked.bool(False),
    yweightPenalty = cms.untracked.double(1.5),
    hitDropLimit5Hits = cms.untracked.double(0.8),
    preClustering = cms.untracked.bool(True),
    hitDropLimit4Hits = cms.untracked.double(0.6),
    hitDropLimit6Hits = cms.untracked.double(0.3333),
    maxRecHitsInCluster = cms.untracked.int32(20),
    CSCDebug = cms.untracked.bool(False),
    onlyBestSegment = cms.untracked.bool(False),
    Pruning = cms.untracked.bool(False),
    dYclusBoxMax = cms.untracked.double(8.0)
)
ST_ME1A = cms.PSet(
    curvePenaltyThreshold = cms.untracked.double(0.85),
    minHitsPerSegment = cms.untracked.int32(3),
    yweightPenaltyThreshold = cms.untracked.double(1.0),
    curvePenalty = cms.untracked.double(2.0),
    dXclusBoxMax = cms.untracked.double(4.0),
    BrutePruning = cms.untracked.bool(False),
    yweightPenalty = cms.untracked.double(1.5),
    hitDropLimit5Hits = cms.untracked.double(0.8),
    preClustering = cms.untracked.bool(True),
    hitDropLimit4Hits = cms.untracked.double(0.6),
    hitDropLimit6Hits = cms.untracked.double(0.3333),
    maxRecHitsInCluster = cms.untracked.int32(24),
    CSCDebug = cms.untracked.bool(False),
    onlyBestSegment = cms.untracked.bool(False),
    Pruning = cms.untracked.bool(False),
    dYclusBoxMax = cms.untracked.double(8.0)
)
CSCSegAlgoST = cms.PSet(
    chamber_types = cms.vstring('ME1/a', 
        'ME1/b', 
        'ME1/2', 
        'ME1/3', 
        'ME2/1', 
        'ME2/2', 
        'ME3/1', 
        'ME3/2', 
        'ME4/1'),
    #    vstring chamber_types = {"ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2"}
    #    vint32 parameters_per_chamber_type = {2, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    algo_name = cms.string('CSCSegAlgoST'),
    algo_psets = cms.VPSet(cms.PSet(
        ST_ME1234
    ), 
        cms.PSet(
            ST_ME1A
        )),
    parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 
        1, 1, 1, 1)
)


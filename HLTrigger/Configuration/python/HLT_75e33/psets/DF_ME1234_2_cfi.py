import FWCore.ParameterSet.Config as cms

DF_ME1234_2 = cms.PSet(
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
)
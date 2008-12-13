import FWCore.ParameterSet.Config as cms

# MuonAlignmentAnalyzer
muonAlignment = cms.EDAnalyzer("MuonAlignment",
    doDT = cms.untracked.bool(True),
    doCSC = cms.untracked.bool(True),
    doSummary = cms.untracked.bool(False),
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string("MuonAlignmentMonitor.root"),
    MuonCollection = cms.InputTag("muons"),
    RecHits4DDTCollectionTag = cms.InputTag("dt4DSegments"),
    RecHits4DCSCCollectionTag = cms.InputTag("cscSegments"),
    resLocalXRangeStation1 = cms.untracked.double(0.1),
    resLocalXRangeStation2 = cms.untracked.double(0.3),
    resLocalXRangeStation3 = cms.untracked.double(1.0),
    resLocalXRangeStation4 = cms.untracked.double(3.0),
    resLocalYRangeStation1 = cms.untracked.double(0.7),
    resLocalYRangeStation2 = cms.untracked.double(1.0),
    resLocalYRangeStation3 = cms.untracked.double(3.0),
    resLocalYRangeStation4 = cms.untracked.double(5.0),
    resThetaRange = cms.untracked.double(0.1),
    resPhiRange = cms.untracked.double(0.1),
    meanPositionRange = cms.untracked.double(0.5),
    rmsPositionRange = cms.untracked.double(0.1),
    meanAngleRange = cms.untracked.double(0.1),
    rmsAngleRange = cms.untracked.double(0.01),
    nbins = cms.untracked.uint32(500),
    min1DTrackRecHitSize = cms.untracked.uint32(1),
    min4DTrackSegmentSize = cms.untracked.uint32(1)
 )




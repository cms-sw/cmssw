import FWCore.ParameterSet.Config as cms

# Put here the modules you want the cfg file to use,
# then include this file in your cfg file.
# i.e. in Validator.cfg replace 'module demo = Validator {} '
# with 'include "anlyzerDir/Validator/data/Validator.cfi" '.
# (Remember that filenames are case sensitive.)

MuonAlignmentMonitor = cms.EDAnalyzer("MuonAlignmentAnalyzer",
    #       To do resolution plots:
    #       untracked string DataType = "SimData"      # needs g4SimHits!!!
    DataType = cms.untracked.string('RealData'),
    
    # range of pt/mass histograms to analyze
    ptRangeMin = cms.untracked.double(0.0),
    ptRangeMax = cms.untracked.double(300.0),
    invMassRangeMin = cms.untracked.double(0.0),
    invMassRangeMax = cms.untracked.double(200.0),

    doSAplots = cms.untracked.bool(True),
    StandAloneTrackCollectionTag = cms.InputTag("globalMuons"),

    doGBplots = cms.untracked.bool(True),
    GlobalMuonTrackCollectionTag = cms.InputTag("standAloneMuons","UpdatedAtVtx"),

    doResplots = cms.untracked.bool(True),
    RecHits4DDTCollectionTag = cms.InputTag("dt4DSegments"),
    RecHits4DCSCCollectionTag = cms.InputTag("cscSegments"),

    #residual range limits: cm and rad
    resLocalXRangeStation1 = cms.untracked.double(0.1),
    resLocalXRangeStation2 = cms.untracked.double(0.3),
    resLocalXRangeStation3 = cms.untracked.double(3.0),
    resLocalXRangeStation4 = cms.untracked.double(3.0),
    resLocalYRangeStation1 = cms.untracked.double(0.7),
    resLocalYRangeStation2 = cms.untracked.double(0.7),
    resLocalYRangeStation3 = cms.untracked.double(5.0),
    resLocalYRangeStation4 = cms.untracked.double(5.0),
    resThetaRange = cms.untracked.double(0.1),
    resPhiRange = cms.untracked.double(0.1),
    nbins = cms.untracked.uint32(500),
    min1DTrackRecHitSize = cms.untracked.uint32(1),
    min4DTrackSegmentSize = cms.untracked.uint32(1)
)


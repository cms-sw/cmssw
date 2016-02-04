import FWCore.ParameterSet.Config as cms

# -*-SH-*-
from RecoMuon.MuonIdentification.isolation_cff import *
from RecoMuon.MuonIdentification.caloCompatibility_cff import *
from RecoMuon.MuonIdentification.MuonTimingFiller_cfi import *
from RecoMuon.MuonIdentification.TrackerKinkFinder_cfi import *
from TrackingTools.TrackAssociator.default_cfi import *
muons1stStep = cms.EDProducer("MuonIdProducer",
    # MuonCaloCompatibility
    MuonCaloCompatibilityBlock,
    # TrackDetectorAssociator
    TrackAssociatorParameterBlock,
    # MuonIsolation
    MIdIsoExtractorPSetBlock,
    # MuonTiming
    TimingFillerBlock,
    # Kink finder
    TrackerKinkFinderParametersBlock,

    fillEnergy = cms.bool(True),
    # OR
    maxAbsPullX = cms.double(4.0),
    maxAbsEta = cms.double(3.0),

    # Selection parameters
    minPt = cms.double(0.5),
    inputCollectionTypes = cms.vstring('inner tracks', 
                                       'links', 
                                       'outer tracks',
                                       'tev firstHit',
                                       'tev picky',
                                       'tev dyt'),
    addExtraSoftMuons = cms.bool(False),
    fillGlobalTrackRefits = cms.bool(True),

    # internal
    debugWithTruthMatching = cms.bool(False),
    # input tracks
    inputCollectionLabels = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("globalMuons"), cms.InputTag("standAloneMuons","UpdatedAtVtx"),
                                          cms.InputTag("tevMuons","firstHit"),cms.InputTag("tevMuons","picky"),cms.InputTag("tevMuons","dyt")),
    fillCaloCompatibility = cms.bool(True),
    # OR
    maxAbsPullY = cms.double(9999.0),
    # AND
    maxAbsDy = cms.double(9999.0),
    minP = cms.double(2.5),
    minPCaloMuon = cms.double(1.0),

    # Match parameters
    maxAbsDx = cms.double(3.0),
    fillIsolation = cms.bool(True),
    writeIsoDeposits = cms.bool(True),
    minNumberOfMatches = cms.int32(1),
    fillMatching = cms.bool(True),

    # global fit for candidate p4 requirements
    ptThresholdToFillCandidateP4WithGlobalFit = cms.double(200.0),
    sigmaThresholdToFillCandidateP4WithGlobalFit = cms.double(2.0),

    # global quality
    fillGlobalTrackQuality = cms.bool(False), #input depends on external module output --> set to True where the sequence is defined
    globalTrackQualityInputTag = cms.InputTag('glbTrackQual'),

    # tracker kink finding
    fillTrackerKink = cms.bool(True),
    
    # calo muons
    minCaloCompatibility = cms.double(0.6),

    # arbitration cleaning                       
    runArbitrationCleaner = cms.bool(True),
    arbitrationCleanerOptions = cms.PSet( ME1a = cms.bool(True),
                                          Overlap = cms.bool(True),
                                          Clustering = cms.bool(True),
                                          OverlapDPhi   = cms.double(0.0786), # 4.5 degrees
                                          OverlapDTheta = cms.double(0.02), # 1.14 degrees
                                          ClusterDPhi   = cms.double(0.6), # 34 degrees
                                          ClusterDTheta = cms.double(0.02) # 1.14
    )
)
                       
muonEcalDetIds = cms.EDProducer("InterestingEcalDetIdProducer",
                                inputCollection = cms.InputTag("muons1stStep")
)



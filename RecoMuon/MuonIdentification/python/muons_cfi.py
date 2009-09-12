import FWCore.ParameterSet.Config as cms

# -*-SH-*-
from RecoMuon.MuonIdentification.isolation_cff import *
from RecoMuon.MuonIdentification.caloCompatibility_cff import *
from RecoMuon.MuonIdentification.MuonTimingFiller_cfi import *
from TrackingTools.TrackAssociator.default_cfi import *
muons = cms.EDProducer("MuonIdProducer",
    # MuonCaloCompatibility
    MuonCaloCompatibilityBlock,
    # TrackDetectorAssociator
    TrackAssociatorParameterBlock,
    # MuonIsolation
    MIdIsoExtractorPSetBlock,
    # MuonTiming
    TimingFillerBlock,
    fillEnergy = cms.bool(True),
    # OR
    maxAbsPullX = cms.double(4.0),
    maxAbsEta = cms.double(3.0),
    #
    # Selection parameters
    minPt = cms.double(1.0),
    inputCollectionTypes = cms.vstring('inner tracks', 
        'links', 
        'outer tracks'),
    addExtraSoftMuons = cms.bool(False),
    #
    # internal
    debugWithTruthMatching = cms.bool(False),
    # input tracks
    inputCollectionLabels = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("globalMuons"), cms.InputTag("standAloneMuons","UpdatedAtVtx")),
    fillCaloCompatibility = cms.bool(True),
    # OR
    maxAbsPullY = cms.double(9999.0),
    # AND
    maxAbsDy = cms.double(9999.0),
    minP = cms.double(3.0),
    #
    # Match parameters
    maxAbsDx = cms.double(3.0),
    fillIsolation = cms.bool(True),
    minNumberOfMatches = cms.int32(1),
    fillMatching = cms.bool(True),
    #
    # global fit for candidate p4 requirements
    ptThresholdToFillCandidateP4WithGlobalFit = cms.double(200.0),
    sigmaThresholdToFillCandidateP4WithGlobalFit = cms.double(2.0),
    # 
    # global quality
    fillGlobalTrackQuality = cms.bool(False), #input depends on external module output --> set to True where the sequence is defined
    globalTrackQualityInputTag = cms.InputTag('glbTrackQual')
)




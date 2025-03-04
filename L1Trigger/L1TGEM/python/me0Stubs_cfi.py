import FWCore.ParameterSet.Config as cms

me0Stubs = cms.EDProducer("ME0StubProducer",
    # parameters for l1t::me0::Config
    skipCentroids = cms.bool(False), 
    layerThresholdPatternId = cms.vint32(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 4), 
    layerThresholdEta = cms.vint32(4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4), 
    maxSpan = cms.int32(37), 
    width = cms.int32(192), 
    deghostPre = cms.bool(True), 
    deghostPost = cms.bool(True), 
    groupWidth = cms.int32(8), 
    ghostWidth = cms.int32(1), 
    xPartitionEnabled = cms.bool(True), 
    enableNonPointing = cms.bool(False), 
    crossPartitionSegmentWidth = cms.int32(4), 
    numOutputs = cms.int32(4), 
    checkIds = cms.bool(False), 
    edgeDistance = cms.int32(2), 
    numOr = cms.int32(2),
    mseThreshold = cms.double(0.75),
    # input collections : GEMPadDigis
    InputCollection = cms.InputTag("GEMPadDigis"),
)
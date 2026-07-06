import FWCore.ParameterSet.Config as cms

me0Stubs = cms.EDProducer("ME0TriggerProducerV2",
    # parameters for l1t::me0::Config
    skipCentroids = cms.bool(False), 
    layerThresholdPatternId = cms.vint32(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 4), 
    layerThresholdEta = cms.vint32(4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4), 
    maxSpan = cms.int32(37), 
    width = cms.int32(192), 
    deghostPre = cms.bool(True), 
    deghostPost = cms.bool(False), 
    groupWidth = cms.int32(16), 
    ghostWidth = cms.int32(1), 
    xPartitionEnabled = cms.bool(True), 
    enableNonPointing = cms.bool(False), 
    crossPartitionSegmentWidth = cms.int32(4), 
    clearanceWidth = cms.int32(0), 
    numOutputs = cms.int32(8), 
    checkIds = cms.bool(False), 
    edgeDistance = cms.int32(2), 
    numOr = cms.int32(2),
    mseThreshold = cms.double(0.75),
    bendAngleCut = cms.double(1.0), 
    BXWindow = cms.int32(3),
    enablePeaking = cms.bool(True),
    debug = cms.bool(False),
    # input collections : GEMPadDigis
    ME0PadDigis = cms.InputTag("GEMPadDigis"),
)
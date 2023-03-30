import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizerConditionsESProducer_cfi import *

DefaultClusterizer = cms.PSet(
    Algorithm = cms.string('ThreeThresholdAlgorithm'),
    ChannelThreshold = cms.double(2.0),
    SeedThreshold = cms.double(3.0),
    ClusterThreshold = cms.double(5.0),
    MaxSequentialHoles = cms.uint32(0),
    MaxSequentialBad = cms.uint32(1),
    MaxAdjacentBad = cms.uint32(0),
    MaxClusterSize = cms.uint32(768),
    RemoveApvShots     = cms.bool(True),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
    ConditionsLabel = cms.string("")
    )

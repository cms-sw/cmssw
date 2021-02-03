import FWCore.ParameterSet.Config as cms

layerInfo = cms.PSet(
    TEC = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(
            refToPSet_ = cms.string('SiStripClusterChargeCutNone')
        ),
        maxRing = cms.int32(7),
        minRing = cms.int32(6),
        useRingSlector = cms.bool(False)
    ),
    TOB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(
            refToPSet_ = cms.string('SiStripClusterChargeCutNone')
        )
    )
)
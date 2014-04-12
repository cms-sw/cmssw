import FWCore.ParameterSet.Config as cms

DummyMetricsUpdator = cms.PSet(
    MetricsUpdatorName = cms.string( "DummyMetricsUpdator" ),
    FixedAlignableDetIds = cms.untracked.vuint32()
)


PixelTrackerMetricsUpdator = cms.PSet(
    MetricsUpdatorName = cms.string( "SimpleMetricsUpdator" ),
    MaxMetricsDistance = cms.untracked.int32(2)
)


OuterTrackerMetricsUpdator = cms.PSet(
    MetricsUpdatorName = cms.string( "SimpleMetricsUpdator" ),
    MaxMetricsDistance = cms.untracked.int32(1)
)


InnerTrackerExtendedMetricsUpdator = cms.PSet(
    MetricsUpdatorName = cms.string( "SimpleMetricsUpdator" ),
    MaxMetricsDistance = cms.untracked.int32(3),

    ApplyAdditionalSelectionCriterion = cms.untracked.bool( True ),
    MinDeltaPerp = cms.double(-5.0),
    MaxDeltaPerp = cms.double(15.0),
    MinDeltaZ = cms.double(-5.0),
    MaxDeltaZ = cms.double(20.0),
    GeomDist = cms.double(20.0),
    MetricalThreshold = cms.uint32(1)
)


OuterTrackerExtendedMetricsUpdator = cms.PSet(
    MetricsUpdatorName = cms.string( "SimpleMetricsUpdator" ),
    MaxMetricsDistance = cms.untracked.int32(2),
    
    ApplyAdditionalSelectionCriterion = cms.untracked.bool( True ),
    MinDeltaPerp = cms.double(-5.0),
    MaxDeltaPerp = cms.double(20.0),
    MinDeltaZ = cms.double(-5.0),
    MaxDeltaZ = cms.double(40.0),
    GeomDist = cms.double(30.0),
    MetricalThreshold = cms.uint32(1)
)


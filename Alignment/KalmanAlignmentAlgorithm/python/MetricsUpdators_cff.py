import FWCore.ParameterSet.Config as cms

DummyMetricsUpdator = cms.PSet(
    FixedAlignableDetIds = cms.untracked.vuint32(),
    MetricsUpdatorName = cms.string('DummyMetricsUpdator')
)
SimpleMetricsUpdator = cms.PSet(
    MetricsUpdatorName = cms.string('SimpleMetricsUpdator'),
    MaxMetricsDistance = cms.untracked.int32(5)
)


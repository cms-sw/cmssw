import FWCore.ParameterSet.Config as cms

TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer",
  usePhase2Stacks = cms.bool(False)
)

from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
vectorHits.toModify(TrackerRecoGeometryESProducer, usePhase2Stacks = True)


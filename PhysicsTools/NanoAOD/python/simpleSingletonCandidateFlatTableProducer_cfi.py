import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer as default

simpleSingletonCandidateFlatTableProducer = default.clone(
  singleton = cms.bool(True),
)
del simpleSingletonCandidateFlatTableProducer.cut

import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input=10

from GeneratorInterface.Core.ExternalGeneratorFilter import *
process.generator = ExternalGeneratorFilter(
    cms.EDFilter("FailingGeneratorFilter",
                 failAt=cms.int32(int(sys.argv[1])),
                 failureType = cms.int32(int(sys.argv[2]))),
    _external_process_waitTime_ = cms.untracked.uint32(5),
    _external_process_verbose_ = cms.untracked.bool(True),
    _external_process_components_ =cms.vstring()
)

process.p = cms.Path(process.generator)

process.add_(cms.Service("RandomNumberGeneratorService",
  generator = cms.PSet(
      initialSeed = cms.untracked.uint32(123),
      engineName = cms.untracked.string('HepJamesRandom')
  )
))

import FWCore.ParameterSet.Config as cms
import os

mem_limit = 1.0 if not '_UBSAN_X' in os.getenv('CMSSW_VERSION', '') else 1.5

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.add_(cms.Service("ResourceEnforcer",
                         maxVSize = cms.untracked.double(mem_limit),
                         maxRSS = cms.untracked.double(mem_limit),
                         maxTime = cms.untracked.double(1.0)))

process.thing = cms.EDProducer("ThingProducer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.p = cms.Path(process.thing)


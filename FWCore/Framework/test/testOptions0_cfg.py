# Test of options when all parameters taken from Config.py
import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.options.dumpOptions = True

# This configuration file tests the code that checks
# for missing ROOT dictionaries. It is intentional
# that the cmsRun job fails with an exception.

# Note that this only tests one simple case,
# but one could use this as a starting point
# to test other cases by editing the classes_def.xml
# to comment out dictionary definitions or the editing
# the consumes and produces calls in
# MissingDictionaryTestProducer to test the many other
# possible cases in this part of the code.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testMissingDictionaries.root')
)

process.a3 = cms.EDProducer("TestMod")

process.a1 = cms.EDProducer("MissingDictionaryTestProducer",
  inputTag = cms.InputTag("a2")
)

process.p = cms.Path(process.a3 * process.a1)

process.e = cms.EndPath(process.out)

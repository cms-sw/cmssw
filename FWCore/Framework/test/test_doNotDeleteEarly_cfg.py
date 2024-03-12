import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(3))

process.maker = cms.EDProducer("DeleteEarlyProducer")

process.reader = cms.EDAnalyzer("DeleteEarlyReader",tag = cms.untracked.InputTag("maker") )

process.tester = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
                                expectedValues = cms.untracked.vuint32(1,3,5))

process.p = cms.Path(process.maker+process.reader+process.tester)
# foo bar baz
# Ur4GQrRivv1Ow
# 9XiR1EILS9iMn

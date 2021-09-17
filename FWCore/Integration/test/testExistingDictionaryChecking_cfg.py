import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.prod = cms.EDProducer("ExistingDictionaryTestProducer")
process.read = cms.EDAnalyzer("ExistingDictionaryTestAnalyzer",
                              src = cms.InputTag("prod")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testExistingDictionaryChecking.root'),
    outputCommands = cms.untracked.vstring(
        'keep *_prod_*_*',
    )
)

process.p = cms.Path(process.prod+process.read)
process.ep = cms.EndPath(process.out)

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.intvec = cms.EDProducer("IntVectorProducer",
    count = cms.int32(9),
    ivalue = cms.int32(11),
    delta = cms.int32(1)
)

process.associationMapProducer = cms.EDProducer("AssociationMapProducer",
    inputTag = cms.InputTag("intvec")
)

process.test = cms.EDAnalyzer("AssociationMapAnalyzer",
    inputTag = cms.InputTag("intvec"),
    associationMapTag = cms.InputTag("associationMapProducer")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('AssociationMapTest.root')
)

process.p = cms.Path(process.intvec * process.associationMapProducer * process.test)

process.e = cms.EndPath(process.out)

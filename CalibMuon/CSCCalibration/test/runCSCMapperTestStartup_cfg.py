import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.dummy = cms.ESSource("EmptyESSource",
    recordName = cms.string("CSCChannelMapperRecord"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process.CSCChannelMapperESProducer = cms.ESProducer("CSCChannelMapperESProducer", AlgoName = cms.string("CSCChannelMapperStartup") )

process.analyze = cms.EDAnalyzer("CSCMapperTestStartup")

process.test = cms.Path(process.analyze)


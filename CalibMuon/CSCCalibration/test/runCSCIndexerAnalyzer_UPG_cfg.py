import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.dummy = cms.ESSource("EmptyESSource",
    recordName = cms.string("CSCIndexerRecord"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

##process.load("CalibMuon.CSCCalibration.CSCIndexer_cfi")
## The above cfi gives rise to the following line:

##process.CSCIndexerESProducer = cms.ESProducer("CSCIndexerESProducer", AlgoName = cms.string("CSCIndexerStartup") )
process.CSCIndexerESProducer = cms.ESProducer("CSCIndexerESProducer", AlgoName = cms.string("CSCIndexerPostls1") )

process.analyze = cms.EDAnalyzer("CSCIndexerAnalyzer")

process.test = cms.Path(process.analyze)


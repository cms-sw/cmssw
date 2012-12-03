import FWCore.ParameterSet.Config as cms

process.dummy1 = cms.ESSource("EmptyESSource",
 recordName = cms.string("CSCIndexerRecord"),
 firstValid = cms.vuint32(1),
 iovIsRunNotTime = cms.bool(True)  )

CSCIndexerESProducer = cms.ESProducer("CSCIndexerESProducer",
  AlgoName = cms.string("CSCIndexerStartup")
)


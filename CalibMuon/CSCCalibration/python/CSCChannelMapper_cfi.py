import FWCore.ParameterSet.Config as cms

process.dummy2 = cms.ESSource("EmptyESSource",
 recordName = cms.string("CSCChannelMapperRecord"),
 firstValid = cms.vuint32(1),
 iovIsRunNotTime = cms.bool(True)  )

CSCChannelMapperESProducer = cms.ESProducer("CSCChannelMapperESProducer",
  AlgoName = cms.string("CSCChannelMapperStartup")
)


import FWCore.ParameterSet.Config as cms

CSCChannelMapperESSource = cms.ESSource("EmptyESSource",
 recordName = cms.string("CSCChannelMapperRecord"),
 firstValid = cms.vuint32(1),
 iovIsRunNotTime = cms.bool(True)  )

CSCChannelMapperESProducer = cms.ESProducer("CSCChannelMapperESProducer",
  AlgoName = cms.string("CSCChannelMapperStartup")
)

#
# Modify for running in run 2
#
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( CSCChannelMapperESProducer, AlgoName=cms.string("CSCChannelMapperPostls1") )

import FWCore.ParameterSet.Config as cms

CSCIndexerESSource = cms.ESSource("EmptyESSource",
 recordName = cms.string("CSCIndexerRecord"),
 firstValid = cms.vuint32(1),
 iovIsRunNotTime = cms.bool(True)  )

CSCIndexerESProducer = cms.ESProducer("CSCIndexerESProducer",
  AlgoName = cms.string("CSCIndexerStartup")
)

#
# Modify for running in run 2
#
from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify( CSCIndexerESProducer, AlgoName=cms.string("CSCIndexerPostls1") )

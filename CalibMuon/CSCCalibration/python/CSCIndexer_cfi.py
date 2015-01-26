import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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
eras.run2.toModify( CSCIndexerESProducer, AlgoName=cms.string("CSCIndexerPostls1") )

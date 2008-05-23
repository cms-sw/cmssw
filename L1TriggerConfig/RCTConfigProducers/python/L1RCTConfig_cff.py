import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.EcalTPGTools.ecalTPGScale_cff import *

from L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cfi import *

l1RctMaskRcds = cms.ESSource("EmptyESSource",
                             recordname = cms.string('L1RCTChannelMaskRcd'),
                             iovIsRunNotTime = cms.bool(True),
                             firstValid = cms.vuint32(1)
                             )

l1RctParamsRecords = cms.ESSource("EmptyESSource",
                                  recordName = cms.string('L1RCTParametersRcd'),
                                  iovIsRunNotTime = cms.bool(True),
                                  firstValid = cms.vuint32(1)
                                  )

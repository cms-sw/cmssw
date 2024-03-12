import FWCore.ParameterSet.Config as cms

# ECAL and HCAL scales
from L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff import *
from L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cfi import *
# RCT parameters
l1RctParamsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RCTParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# RCT channel mask
l1RctMaskRcds = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RCTChannelMaskRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)



# foo bar baz
# tAFFRuPm6OyyE

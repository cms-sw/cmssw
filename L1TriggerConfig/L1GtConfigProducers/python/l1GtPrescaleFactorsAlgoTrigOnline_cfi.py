#
import FWCore.ParameterSet.Config as cms

l1GtPrescaleFactorsAlgoTrigOnline = cms.ESProducer("L1GtPrescaleFactorsAlgoTrigConfigOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)

# foo bar baz
# nur0zXwHvKfKg

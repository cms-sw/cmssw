#
import FWCore.ParameterSet.Config as cms

l1GtTriggerMenuOnline = cms.ESProducer("L1GtTriggerMenuConfigOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)

# foo bar baz
# 6tQeLwXGg6s8M
# p8mMkMRbilVf5

import FWCore.ParameterSet.Config as cms
L1CaloEcalScaleConfigOnline = cms.ESProducer("L1CaloEcalScaleConfigOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
# foo bar baz
# yglXgDW0QdAzb
# 1sd2IGIn0BpRi

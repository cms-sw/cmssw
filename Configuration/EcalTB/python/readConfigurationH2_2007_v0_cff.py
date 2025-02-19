import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
ecalConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_online')
    ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroups_prelimininary')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            tag = cms.string('EcalTBWeights_preliminary')
        )),
    connect = cms.string('frontier://Frontier/CMS_ECALHCAL_H2_COND_2007'), ##Frontier/CMS_ECALHCAL_H2_COND_2007"

    siteLocalConfig = cms.untracked.bool(True)
)

h2007Trivial = cms.ESSource("EcalTrivialConditionRetriever",
    producedEcalPedestals = cms.untracked.bool(False),
    producedEcalWeights = cms.untracked.bool(False),
    producedEcalIntercalibConstants = cms.untracked.bool(True),
    producedEcalLaserCorrection = cms.untracked.bool(True),
    producedEcalGainRatios = cms.untracked.bool(True),
    producedEcalADCToGeVConstant = cms.untracked.bool(True)
)



import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
ecalDBConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_online')
    ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroups_preliminary')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            tag = cms.string('EcalTBWeights_preliminary')
        )),
    connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ECAL_H4'), ##Frontier/CMS_ECAL_H4_COND_2007" 

    siteLocalConfig = cms.untracked.bool(True)
)

h42007Trivial = cms.ESSource("EcalTrivialConditionRetriever",
    producedEcalPedestals = cms.untracked.bool(False),
    producedEcalWeights = cms.untracked.bool(False),
    producedEcalIntercalibConstants = cms.untracked.bool(True),
    producedEcalLaserCorrection = cms.untracked.bool(True),
    producedEcalGainRatios = cms.untracked.bool(True),
    producedEcalADCToGeVConstant = cms.untracked.bool(True)
)



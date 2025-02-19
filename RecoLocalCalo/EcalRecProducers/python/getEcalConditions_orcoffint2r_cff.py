import FWCore.ParameterSet.Config as cms

ecalConditions = cms.ESSource("PoolDBESSource",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_trivial')
    ), 
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            tag = cms.string('EcalGainRatios_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            tag = cms.string('EcalIntercalibConstants_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroups_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalTBWeightsRcd'),
            tag = cms.string('EcalTBWeights_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalLaserAlphasRcd'),
            tag = cms.string('EcalLaserAlphas_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalLaserAPDPNRatiosRcd'),
            tag = cms.string('EcalLaserAPDPNRatios_trivial')
        ), 
        cms.PSet(
            record = cms.string('EcalLaserAPDPNRatiosRefRcd')
        )),
    messagelevel = cms.untracked.uint32(2),
    catalog = cms.untracked.string('relationalcatalog_oracle://cms_orcoff_int2r/CMS_COND_GENERAL'), ##cms_orcoff_int2r/CMS_COND_GENERAL"

    timetype = cms.string('runnumber'),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_ECAL'), ##cms_orcoff_int2r/CMS_COND_ECAL"

    authenticationMethod = cms.untracked.uint32(1)
)



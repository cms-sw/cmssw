# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_ECAL"				 
import FWCore.ParameterSet.Config as cms

#
# Ecal  calibrations from Frontier
#
from RecoLocalCalo.EcalRecProducers.getEcalConditions_frontier_cff import *
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
ecalConditions.connect = 'frontier://FrontierDev/CMS_COND_ECAL'
ecalConditions.toGet = cms.VPSet(cms.PSet(
    record = cms.string('EcalPedestalsRcd'),
    tag = cms.string('EcalPedestals_mc')
), 
    cms.PSet(
        record = cms.string('EcalADCToGeVConstantRcd'),
        tag = cms.string('EcalADCToGeVConstant_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalChannelStatusRcd'),
        tag = cms.string('EcalChannelStatus_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalGainRatiosRcd'),
        tag = cms.string('EcalGainRatios_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        tag = cms.string('EcalIntercalibConstants_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalIntercalibErrorsRcd'),
        tag = cms.string('EcalIntercalibErrors_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalWeightXtalGroupsRcd'),
        tag = cms.string('EcalWeightXtalGroups_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalTBWeightsRcd'),
        tag = cms.string('EcalTBWeights_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalLaserAlphasRcd'),
        tag = cms.string('EcalLaserAlphas_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalLaserAPDPNRatiosRcd'),
        tag = cms.string('EcalLaserAPDPNRatios_mc')
    ), 
    cms.PSet(
        record = cms.string('EcalLaserAPDPNRatiosRefRcd'),
        tag = cms.string('EcalLaserAPDPNRatiosRef_mc')
    ))


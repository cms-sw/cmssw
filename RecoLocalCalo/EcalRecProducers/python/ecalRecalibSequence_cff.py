import FWCore.ParameterSet.Config as cms

#ECAL conditions
from RecoLocalCalo.EcalRecProducers.getEcalConditions_frontier_cff import *
#ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalRecalibRecHit_cfi import *
ecalRecalibSequence = cms.Sequence(cms.SequencePlaceholder("ecalRecalibRecHit"))
ecalConditions.toGet = cms.VPSet(
    cms.PSet(
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
        record = cms.string('EcalLaserAPDPNRatiosRefRcd'),
        tag = cms.string('EcalLaserAPDPNRatiosRef_trivial')
    )
)

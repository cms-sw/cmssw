import FWCore.ParameterSet.Config as cms

#
# producer to apply corrections to photons
#
correctedPhotons = cms.EDProducer("PhotonCorrectionProducer",
    algoCollection = cms.string('E9ESCPtdr E1E9Ptdr EtaPtdr PhiPtdr'),
    barrelClusterShapeMapProducer = cms.string('hybridSuperClusters'),
    photonCorrCollection = cms.string('correctedPhotonsWithConversions'),
    barrelClusterShapeMapCollection = cms.string('hybridShapeAssoc'),
    endcapClusterShapeMapProducer = cms.string('islandBasicClusters'),
    photonCollection = cms.string(''),
    conversionProducer = cms.string('conversions'),
    conversionCollection = cms.string('conversions'),
    endcapClusterShapeMapCollection = cms.string('islandEndcapShapeAssoc'),
    photonProducer = cms.string('photons')
)



import FWCore.ParameterSet.Config as cms

import RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi

dynamicHybridSuperClusters = RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi.cleanedHybridSuperClusters.clone(
    shapeAssociation = cms.string('dynamicHybridShapeAssoc'),
    dynamicPhiRoad = cms.bool(True),
    basicclusterCollection = cms.string(''),
    dynamicEThresh = cms.bool(True),
    bremRecoveryPset = cms.PSet(
    barrel = cms.PSet(
    cryVec = cms.vint32(17, 15, 13, 12, 11,
                        10, 9, 8, 7, 6),
    cryMin = cms.int32(5),
    etVec = cms.vdouble(5.0, 10.0, 15.0, 20.0, 30.0,
                        40.0, 45.0, 135.0, 195.0, 225.0)
    ),
    endcap = cms.PSet(
    a = cms.double(47.85),
    c = cms.double(0.1201),
    b = cms.double(108.8)
    )
     )
    )

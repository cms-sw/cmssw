import FWCore.ParameterSet.Config as cms

multi5x5SuperClustersCleaned = cms.EDProducer("Multi5x5SuperClusterProducer",
    barrelClusterTag = cms.InputTag("multi5x5BasicClusters","multi5x5BarrelBasicClustersCleaned"),
    barrelEtaSearchRoad = cms.double(0.06),
    barrelPhiSearchRoad = cms.double(0.8),
    barrelSuperclusterCollection = cms.string('multi5x5BarrelSuperClusters'),
    bremRecoveryPset = cms.PSet(
        barrel = cms.PSet(
            cryMin = cms.int32(2),
            cryVec = cms.vint32(
                16, 13, 11, 10, 9,
                8, 7, 6, 5, 4,
                3
            ),
            etVec = cms.vdouble(
                5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 45.0, 55.0, 135.0, 195.0,
                225.0
            )
        ),
        endcap = cms.PSet(
            a = cms.double(47.85),
            b = cms.double(108.8),
            c = cms.double(0.1201)
        )
    ),
    doBarrel = cms.bool(False),
    doEndcaps = cms.bool(True),
    dynamicPhiRoad = cms.bool(False),
    endcapClusterTag = cms.InputTag("multi5x5BasicClustersCleaned","multi5x5EndcapBasicClusters"),
    endcapEtaSearchRoad = cms.double(0.14),
    endcapPhiSearchRoad = cms.double(0.6),
    endcapSuperclusterCollection = cms.string('multi5x5EndcapSuperClusters'),
    seedTransverseEnergyThreshold = cms.double(1.0)
)

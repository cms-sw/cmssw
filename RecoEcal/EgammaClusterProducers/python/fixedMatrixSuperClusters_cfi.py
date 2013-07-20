import FWCore.ParameterSet.Config as cms

#
# $Id: fixedMatrixSuperClusters_cfi.py,v 1.2 2008/04/21 03:24:08 rpw Exp $
#
# Fixed Matrix SuperCluster producer
fixedMatrixSuperClusters = cms.EDProducer("FixedMatrixSuperClusterProducer",
    barrelSuperclusterCollection = cms.string('fixedMatrixBarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterCollection = cms.string('fixedMatrixBarrelBasicClusters'),
    dynamicPhiRoad = cms.bool(True),
    endcapClusterProducer = cms.string('fixedMatrixBasicClusters'),
    barrelPhiSearchRoad = cms.double(0.8),
    endcapPhiSearchRoad = cms.double(0.6),
    VerbosityLevel = cms.string('ERROR'),
    seedTransverseEnergyThreshold = cms.double(1.0),
    doBarrel = cms.bool(False),
    endcapSuperclusterCollection = cms.string('fixedMatrixEndcapSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.06),
    # for brem recovery
    bremRecoveryPset = cms.PSet(
        barrel = cms.PSet(
            cryVec = cms.vint32(16, 13, 11, 10, 9, 
                8, 7, 6, 5, 4, 
                3),
            cryMin = cms.int32(2),
            etVec = cms.vdouble(5.0, 10.0, 15.0, 20.0, 30.0, 
                40.0, 45.0, 55.0, 135.0, 195.0, 
                225.0)
        ),
        endcap = cms.PSet(
            a = cms.double(47.85),
            c = cms.double(0.1201),
            b = cms.double(108.8)
        )
    ),
    doEndcaps = cms.bool(True),
    endcapClusterCollection = cms.string('fixedMatrixEndcapBasicClusters'),
    barrelClusterProducer = cms.string('fixedMatrixBasicClusters')
)



import FWCore.ParameterSet.Config as cms

# The HGCAL components, in the order of the merged layer clusters: EE, HSci, HSi
hgcal_components = ["", "HSci", "HSi"]

# The legacy layer clusters of the HGCAL components, converted from their SoA layer clusters into one merged
# collection in one step
hltMergeLayerClustersL1Seeded = cms.EDProducer("HGCalMergedLayerClustersFromSoAProducer",
    layerClustersSoA = cms.VInputTag(*["hltHgcalSoALayerClustersProducer" + x + "L1Seeded" for x in hgcal_components]),
    hgcalRecHitsSoA = cms.VInputTag(*["hltHgcalSoARecHitsProducer" + x + "L1Seeded" for x in hgcal_components]),
    hgcalRecHitsLayerClustersSoA = cms.VInputTag(*["hltHgcalSoARecHitsLayerClustersProducer" + x + "L1Seeded" for x in hgcal_components]),
    detectors = cms.vstring("EE", "BH", "FH"),
    extraLayerClusters = cms.VInputTag(),
    extraTimeLayerClusters = cms.VInputTag(),
    timeClname = cms.string("timeLayerCluster")
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

# with the barrel, its layer clusters come after the HGCAL ones
ticl_barrel.toModify(hltMergeLayerClustersL1Seeded,
    extraLayerClusters = ["hltBarrelLayerClustersEBL1Seeded"],
    extraTimeLayerClusters = ["hltBarrelLayerClustersEBL1Seeded:timeLayerCluster"]
)

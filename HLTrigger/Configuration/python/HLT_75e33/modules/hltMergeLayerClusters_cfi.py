import FWCore.ParameterSet.Config as cms

# The HGCAL components, in the order of the merged layer clusters: EE, HSci, HSi
hgcal_components = ["", "HSci", "HSi"]
hgcal_detectors = ["EE", "BH", "FH"]

barrel_layerClusters = [
    "hltBarrelLayerClustersEB",
    "hltBarrelLayerClustersHB"
]
barrel_time_layerClusters = [x + ":timeLayerCluster" for x in barrel_layerClusters]

# The legacy layer clusters of the HGCAL components, converted from their SoA layer clusters into one merged
# collection in one step
hltMergeLayerClusters = cms.EDProducer("HGCalMergedLayerClustersFromSoAProducer",
    layerClustersSoA = cms.VInputTag(*["hltHgcalSoALayerClustersProducer" + x for x in hgcal_components]),
    hgcalRecHitsSoA = cms.VInputTag(*["hltHgcalSoARecHitsProducer" + x for x in hgcal_components]),
    hgcalRecHitsLayerClustersSoA = cms.VInputTag(*["hltHgcalSoARecHitsLayerClustersProducer" + x for x in hgcal_components]),
    detectors = cms.vstring(*hgcal_detectors),
    extraLayerClusters = cms.VInputTag(),
    extraTimeLayerClusters = cms.VInputTag(),
    timeClname = cms.string("timeLayerCluster")
)

hltMergeLayerClustersSerialSync = cms.EDProducer("HGCalMergedLayerClustersFromSoAProducer",
    layerClustersSoA = cms.VInputTag(*["hltHgcalSoALayerClustersProducer" + x + "SerialSync" for x in hgcal_components]),
    hgcalRecHitsSoA = cms.VInputTag(*["hltHgcalSoARecHitsProducer" + x + "SerialSync" for x in hgcal_components]),
    hgcalRecHitsLayerClustersSoA = cms.VInputTag(*["hltHgcalSoARecHitsLayerClustersProducer" + x + "SerialSync" for x in hgcal_components]),
    detectors = cms.vstring(*hgcal_detectors),
    extraLayerClusters = cms.VInputTag(),
    extraTimeLayerClusters = cms.VInputTag(),
    timeClname = cms.string("timeLayerCluster")
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

# with the barrel, its layer clusters come after the HGCAL ones
ticl_barrel.toModify(hltMergeLayerClusters,
    extraLayerClusters = barrel_layerClusters,
    extraTimeLayerClusters = barrel_time_layerClusters
)

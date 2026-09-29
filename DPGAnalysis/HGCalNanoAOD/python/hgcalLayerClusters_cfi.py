import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata

hgcalLayerClustersTable = cms.EDProducer(
    "LayerClustersCollectionTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("hgcalMergeLayerClusters"),
    cut=cms.string(""),
    name=cms.string("HGCalLayerClusters"),
    doc=cms.string("Offline HGCAL Layer Clusters"),
    singleton=cms.bool(False),
    variables=cms.PSet(
        energy=Var("energy", "float",
                   doc="Energy of the LayerCluster [GeV]"),
        correctedEnergy=Var("correctedEnergy", "float",
                            doc="Corrected energy of the LayerCluster [GeV]"),
        position_x=Var(
            "x", "float", doc="X coordinate of the LayerCluster [cm]"),
        position_y=Var(
            "y", "float", doc="Y coordinate of the LayerCluster [cm]"),
        position_z=Var(
            "z", "float", doc="Z coordinate of the LayerCluster [cm]"),
        position_eta=Var(
            "eta", "float", doc="eta coordinate of the LayerCluster"),
        position_phi=Var(
            "phi", "float", doc="phi coordinate of the LayerCluster"),
        nHits=Var(
            "size", "int", doc="Number of RecHits in LayerCluster"),
        algoID=Var(
            "algoID", "int", doc="ID of the algo used for producing LayerCluster"),
    ),
)

# Extra table for layer cluster timing information
hgcalLayerClustersExtraTable = cms.EDProducer(
    "LayerClustersExtraTableProducer",
    tableName=cms.string("HGCalLayerClusters"),
    skipNonExistingSrc=cms.bool(True),
    time_layerclusters=cms.InputTag("hgcalMergeLayerClusters", "timeLayerCluster"),
    precision=cms.int32(7)
)

# Sequence for layer clusters tables
hgcalLayerClustersTableSequence = cms.Sequence(
    hgcalLayerClustersTable + hgcalLayerClustersExtraTable
)

import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata
from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels

# Tracksters
hltLayerClustersTable = cms.EDProducer(
    "LayerClustersCollectionTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("hltMergeLayerClusters"),
    cut=cms.string(""),
    name=cms.string("hltMergeLayerClusters"),
    doc=cms.string("HLT HGCAL Layer Clusters"),
    singleton=cms.bool(False),  # the number of entries is variable
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

hltLayerClustersExtraTable = cms.EDProducer("LayerClustersExtraTableProducer",
    tableName=cms.string("hltMergeLayerClusters"),
    skipNonExistingSrc=cms.bool(True),
    time_layerclusters=cms.InputTag("hltMergeLayerClusters", "timeLayerCluster"),
    precision=cms.int32(7))


hltLayerClustersTableSequence = cms.Sequence(hltLayerClustersTable + hltLayerClustersExtraTable)


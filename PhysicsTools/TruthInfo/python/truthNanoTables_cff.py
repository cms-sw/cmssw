# NanoAOD tables of the calo-boundary training truth: the truth particle table and the
# reco <-> truth association tables for the tracksters and the layer clusters, from the
# maps of SimGeneral/TruthGraphAssociatorProducers/truthCaloBoundaryAssociators_cff.
#
# Layout, the one of the TICL reco <-> sim trackster tables so the parquet conversion treats
# them alike: TruthCB (one row per calo-boundary particle), <A> (rows parallel to its
# source collection, with <A>_n<A>Links / <A>_o<A>Links) and the flat <A>Links table with
# index (row in the target table), graphId, sharedEnergy or sharedEnergyFraction, score.
# Every module label starts with ticlTruth so "keep nanoaodFlatTable_ticl*_*_*" keeps them.

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.truthParticleFlatTable_cfi import truthParticleFlatTable
from PhysicsTools.TruthInfo.truthBranchAssociationFlatTable_cfi import truthBranchAssociationFlatTable

ticlTruthCBTable = truthParticleFlatTable.clone(
    targets=cms.InputTag("truthBranchTargetsCaloBoundary", "truthToRecoTargetsCaloBoundary"),
    # Only the boundary particles with sim energy in HGCAL: at PU 30 three targets out of
    # four are pile-up particles that never reached HGCAL and could match nothing.
    requireHGCALHits=True,
    name="TruthCB",
    doc="MC-truth graph particles at the tracker-calorimeter boundary with HGCAL sim energy (signal and in-time pile-up)",
)
# The rows actually written, so a link index is a row of TruthCB whatever the filter.
_targets = cms.InputTag("ticlTruthCBTable", "selected")

ticlTruthAssocCLUE3DHighToTruthCBTable = truthBranchAssociationFlatTable.clone(
    src=cms.InputTag("truthCaloBoundaryTracksterAssociators", "ticlTrackstersCLUE3DHighRecoToTruthFixed"),
    targets=_targets,
    direction="recoToTruth",
    name="CLUE3DHighToTruthCB",
    doc="ticlTrackstersCLUE3DHigh -> TruthCB: every calo-boundary particle sharing energy, best first",
)
ticlTruthAssocTruthCBToCLUE3DHighTable = truthBranchAssociationFlatTable.clone(
    src=cms.InputTag("truthCaloBoundaryTracksterAssociators", "ticlTrackstersCLUE3DHighTruthToReco"),
    targets=_targets,
    direction="truthToReco",
    name="TruthCBToCLUE3DHigh",
    doc="TruthCB -> ticlTrackstersCLUE3DHigh: every trackster sharing energy, best first",
)
ticlTruthAssocLayerClusterToTruthCBTable = truthBranchAssociationFlatTable.clone(
    src=cms.InputTag("truthCaloBoundaryLayerClusterAssociators", "hgcalMergeLayerClustersRecoToTruthFixed"),
    targets=_targets,
    direction="recoToTruth",
    name="LayerClusterToTruthCB",
    doc="hgcalMergeLayerClusters -> TruthCB: every calo-boundary particle sharing energy, best first",
)
ticlTruthAssocTruthCBToLayerClusterTable = truthBranchAssociationFlatTable.clone(
    src=cms.InputTag("truthCaloBoundaryLayerClusterAssociators", "hgcalMergeLayerClustersTruthToReco"),
    targets=_targets,
    direction="truthToReco",
    name="TruthCBToLayerCluster",
    doc="TruthCB -> hgcalMergeLayerClusters: every layer cluster sharing energy, best first",
)

truthNanoTablesTask = cms.Task(
    ticlTruthCBTable,
    ticlTruthAssocCLUE3DHighToTruthCBTable,
    ticlTruthAssocTruthCBToCLUE3DHighTable,
    ticlTruthAssocLayerClusterToTruthCBTable,
    ticlTruthAssocTruthCBToLayerClusterTable,
)
truthNanoTablesSequence = cms.Sequence(truthNanoTablesTask)

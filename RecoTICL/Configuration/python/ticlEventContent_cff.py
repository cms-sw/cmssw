# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""pyTICL-managed TICL Event Content.

The TICL reconstruction ``keep`` statements (the per-iteration tracksters, the
hadronic links, the EGamma superclustering links, the TICL candidate and pfTICL)
are GENERATED from the assembled pyTICL graph: add an iteration or a stage and
its products are kept automatically, with no hand-edited list to drift.  See
:mod:`RecoTICL.Configuration.event_content`.

The non-reconstruction extras are listed explicitly because they are not pyTICL
modules: the sim-truth tracksters and the sim/reco associations (produced by the
validation/sim chain) and the legacy HFNose tracksters (not part of the v5
graph).  ``TICL_RECO/AOD/FEVT/FEVTHLT`` keep the same names and nesting as the
historical ``RecoHGCal_EventContent_cff`` so existing imports are unaffected.
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets
from RecoTICL.Configuration.event_content import keep_statements, RECO

# -- generated: the offline TICL reconstruction products (v5 graph) ---------- #
_recoTICL = keep_statements(presets.v5(), RECO)

# -- explicit extras: not pyTICL reconstruction modules ---------------------- #
# legacy HFNose tracksters (kept for compatibility; not in the v5 graph)
_hfnose = [
    'keep *_ticlTrackstersHFNoseTrkEM_*_*',
    'keep *_ticlTrackstersHFNoseEM_*_*',
    'keep *_ticlTrackstersHFNoseTrk_*_*',
    'keep *_ticlTrackstersHFNoseMIP_*_*',
    'keep *_ticlTrackstersHFNoseHAD_*_*',
    'keep *_ticlTrackstersHFNoseMerge_*_*',
]
# sim-truth + sim/reco associations (validation/sim chain)
_simAssoc = [
    'keep CaloParticles_mix_*_*',
    'keep SimClusters_mix_*_*',
    'keep *_SimClusterToCaloParticleAssociation*_*_*',
    'keep *_layerClusterSimClusterAssociationProducer_*_*',
    'keep *_layerClusterCaloParticleAssociationProducer_*_*',
    'keep *_layerClusterSimTracksterAssociationProducer_*_*',
    'keep *_allTrackstersToSimTrackstersAssociations*_*_*',
]
_simFEVT = [
    'keep *_ticlSimTracksters_*_*',
    'keep *_ticlSimTICLCandidates_*_*',
    'keep *_ticlSimTrackstersFromCP_*_*',
    'keep *_SimTau*_*_*',
    'keep *_allTrackstersToSimTrackstersAssociations*_*_*',
]
# HLT TICL products
_hltTICL = [
    'keep *_hltPfTICL_*_*',
    'keep *_hltTiclTrackstersCLUE3D*_*_*',
    'keep *_hltTiclTracksterLinks*_*_*',
    'keep *_hltTiclCandidate_*_*',
]

# -- EventContent blocks (names/nesting preserved) --------------------------- #
TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

# RECO content - generated TICL reconstruction + sim/association extras
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(_recoTICL + _hfnose + _simAssoc)
)
TICL_RECO.outputCommands.extend(TICL_AOD.outputCommands)

# FEVT content - full debug info
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(_simFEVT)
)
TICL_FEVT.outputCommands.extend(TICL_RECO.outputCommands)

# HLT content
TICL_FEVTHLT = cms.PSet(
    outputCommands = cms.untracked.vstring(_hltTICL)
)
TICL_FEVTHLT.outputCommands.extend(TICL_FEVT.outputCommands)

"""
Scouting to MiniAOD conversion configuration.

This module converts Run3 scouting data to MiniAOD-compatible format with
standard collection names for downstream processing with standard NanoAOD.

Output collections (matching standard MiniAOD names):
- packedPFCandidates: pat::PackedCandidateCollection from scouting particles
- offlineSlimmedPrimaryVertices: reco::VertexCollection from scouting vertices
- slimmedMuons: pat::MuonCollection from scouting muons
- slimmedMuonsNoVtx: pat::MuonCollection from displaced scouting muons (2024+)
- slimmedElectrons: pat::ElectronCollection from scouting electrons
- slimmedPhotons: pat::PhotonCollection from scouting photons
- slimmedJets: pat::JetCollection from scouting PF jets
- slimmedMETs: pat::METCollection computed from scouting PF candidates
- fixedGridRhoAll: double for pileup density
- fixedGridRhoFastjetAll: double for fastjet rho
- gtStage2Digis: L1 trigger unpacking from raw FED data
- scoutingTracks: reco::TrackCollection from scouting tracks

Usage:
    from PhysicsTools.PatFromScouting.scoutingMiniAOD_cff import scoutingMiniAODTask
    process.scoutingMiniAODPath = cms.Path()
    process.scoutingMiniAODPath.associate(scoutingMiniAODTask)

Or via cmsDriver customization:
    --customise PhysicsTools/PatFromScouting/scoutingMiniAOD_cff.customiseForScoutingMiniAOD
"""

import FWCore.ParameterSet.Config as cms

# ============================================================
# Primary Vertices (must be produced before PF candidates)
# ============================================================

offlineSlimmedPrimaryVertices = cms.EDProducer("Run3ScoutingVertexToRecoVertexProducer",
    src = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx")
)

# ============================================================
# PF Candidates from scouting particles
# ============================================================

# Using pat::PackedCandidate for MiniAOD compatibility
# Requires vertices to be produced first for proper vertex references
packedPFCandidates = cms.EDProducer("Run3ScoutingParticleToPackedCandidateProducer",
    src = cms.InputTag("hltScoutingPFPacker"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    CHS = cms.bool(False)
)

# ============================================================
# Muons
# ============================================================

slimmedMuons = cms.EDProducer("PatFromScoutingMuonProducer",
    src = cms.InputTag("hltScoutingMuonPacker")
)

# Displaced muons (no vertex constraint) - available from 2024+
slimmedMuonsNoVtx = cms.EDProducer("PatFromScoutingMuonProducer",
    src = cms.InputTag("hltScoutingMuonPackerNoVtx")
)

# ============================================================
# Electrons
# ============================================================

slimmedElectrons = cms.EDProducer("PatFromScoutingElectronProducer",
    src = cms.InputTag("hltScoutingEgammaPacker")
)

# ============================================================
# Photons
# ============================================================

slimmedPhotons = cms.EDProducer("PatFromScoutingPhotonProducer",
    src = cms.InputTag("hltScoutingEgammaPacker")
)

# ============================================================
# Jets
# ============================================================

slimmedJets = cms.EDProducer("PatFromScoutingJetProducer",
    src = cms.InputTag("hltScoutingPFPacker"),
    pfCandidates = cms.InputTag("packedPFCandidates")
)

# ============================================================
# MET
# ============================================================

slimmedMETs = cms.EDProducer("Run3ScoutingMETProducer",
    metPt = cms.InputTag("hltScoutingPFPacker", "pfMetPt"),
    metPhi = cms.InputTag("hltScoutingPFPacker", "pfMetPhi")
)

# ============================================================
# Tracks
# ============================================================

scoutingTracks = cms.EDProducer("Run3ScoutingTrackToRecoTrackProducer",
    src = cms.InputTag("hltScoutingTrackPacker")
)

# ============================================================
# Beam Spot (from conditions database)
# ============================================================

# BeamSpotProducer reads from conditions - works with GlobalTag
offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

# ============================================================
# Pileup Density (rho) - all variants needed by NanoAOD
# ============================================================

# All eta range
fixedGridRhoAll = cms.EDProducer("FixedGridRhoProducer",
    pfCandidatesTag = cms.InputTag("packedPFCandidates"),
    EtaRegion = cms.string("All")
)

fixedGridRhoFastjetAll = cms.EDProducer("FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag("packedPFCandidates"),
    maxRapidity = cms.double(5.0),
    gridSpacing = cms.double(0.55)
)

# Central region (|eta| < 2.5)
fixedGridRhoFastjetCentral = cms.EDProducer("FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag("packedPFCandidates"),
    maxRapidity = cms.double(2.5),
    gridSpacing = cms.double(0.55)
)

# Central "calo-like" - use PF candidates in central region
# (scouting doesn't have calo towers, so we approximate with PF)
fixedGridRhoFastjetCentralCalo = cms.EDProducer("FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag("packedPFCandidates"),
    maxRapidity = cms.double(2.5),
    gridSpacing = cms.double(0.55)
)

# Central charged-only (approximate - uses all PF candidates for scouting)
fixedGridRhoFastjetCentralChargedPileUp = cms.EDProducer("FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag("packedPFCandidates"),
    maxRapidity = cms.double(2.5),
    gridSpacing = cms.double(0.55)
)

# Neutral-only variant (approximate - uses all PF candidates for scouting)
fixedGridRhoFastjetCentralNeutral = cms.EDProducer("FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag("packedPFCandidates"),
    maxRapidity = cms.double(2.5),
    gridSpacing = cms.double(0.55)
)

# ============================================================
# L1 Trigger Unpacking
# ============================================================

gtStage2Digis = cms.EDProducer("L1TRawToDigi",
    InputLabel = cms.InputTag("hltFEDSelectorL1"),
    Setup = cms.string("stage2::GTSetup"),
    FedIds = cms.vint32(1404),
)

# ============================================================
# Task and Sequence Definitions
# ============================================================

# Core task - objects needed for standard NanoAOD
# Note: slimmedMuonsNoVtx is not included here; it is only available from 2024+
# and should be added explicitly when processing 2024+ data
scoutingMiniAODCoreTask = cms.Task(
    packedPFCandidates,
    offlineSlimmedPrimaryVertices,
    offlineBeamSpot,
    slimmedMuons,
    slimmedElectrons,
    slimmedPhotons,
    slimmedJets,
    slimmedMETs,
    # All rho variants needed by NanoAOD
    fixedGridRhoAll,
    fixedGridRhoFastjetAll,
    fixedGridRhoFastjetCentral,
    fixedGridRhoFastjetCentralCalo,
    fixedGridRhoFastjetCentralChargedPileUp,
    fixedGridRhoFastjetCentralNeutral,
)

# Full task including L1 unpacking and tracks
scoutingMiniAODTask = cms.Task(
    scoutingMiniAODCoreTask,
    gtStage2Digis,
    scoutingTracks,
)

# Sequence for backward compatibility
scoutingMiniAODSequence = cms.Sequence(scoutingMiniAODTask)


# ============================================================
# Customization Functions
# ============================================================

def customiseForScoutingMiniAOD(process):
    """
    Customization function for cmsDriver to add scouting MiniAOD production.

    Usage:
        cmsDriver.py MINI --customise PhysicsTools/PatFromScouting/scoutingMiniAOD_cff.customiseForScoutingMiniAOD
    """

    # Load particle data table (needed for PF candidate producer)
    process.load("SimGeneral.HepPDTESSource.pdt_cfi")

    # Add all producers to the process
    process.packedPFCandidates = packedPFCandidates.clone()
    process.offlineSlimmedPrimaryVertices = offlineSlimmedPrimaryVertices.clone()
    process.slimmedMuons = slimmedMuons.clone()
    process.slimmedElectrons = slimmedElectrons.clone()
    process.slimmedPhotons = slimmedPhotons.clone()
    process.slimmedJets = slimmedJets.clone()
    process.slimmedMETs = slimmedMETs.clone()
    process.scoutingTracks = scoutingTracks.clone()
    process.fixedGridRhoAll = fixedGridRhoAll.clone()
    process.fixedGridRhoFastjetAll = fixedGridRhoFastjetAll.clone()
    process.gtStage2Digis = gtStage2Digis.clone()

    # Create task and path
    process.scoutingMiniAODTask = cms.Task(
        process.packedPFCandidates,
        process.offlineSlimmedPrimaryVertices,
        process.slimmedMuons,
        process.slimmedElectrons,
        process.slimmedPhotons,
        process.slimmedJets,
        process.slimmedMETs,
        process.scoutingTracks,
        process.fixedGridRhoAll,
        process.fixedGridRhoFastjetAll,
        process.gtStage2Digis,
    )

    process.scoutingMiniAOD_step = cms.Path()
    process.scoutingMiniAOD_step.associate(process.scoutingMiniAODTask)

    # Add to schedule if it exists
    if hasattr(process, 'schedule') and process.schedule is not None:
        process.schedule.insert(0, process.scoutingMiniAOD_step)

    return process


def customiseOutputForScoutingMiniAOD(process):
    """
    Customize output module for scouting MiniAOD content.
    """

    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_packedPFCandidates_*_*',
        'keep *_offlineSlimmedPrimaryVertices_*_*',
        'keep *_slimmedMuons_*_*',
        'keep *_slimmedElectrons_*_*',
        'keep *_slimmedPhotons_*_*',
        'keep *_slimmedJets_*_*',
        'keep *_slimmedMETs_*_*',
        'keep *_scoutingTracks_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_fixedGridRhoAll_*_*',
        'keep *_fixedGridRhoFastjetAll_*_*',
        'keep *_gtStage2Digis_*_*',
    )

    # Find and configure output module
    for name in ['MINIAODoutput', 'MINIAODSIMoutput', 'output', 'out']:
        if hasattr(process, name):
            output = getattr(process, name)
            output.outputCommands = outputCommands
            break

    return process

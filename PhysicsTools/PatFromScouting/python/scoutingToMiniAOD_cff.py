import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run3_scouting_2024_cff import run3_scouting_2024

# Use standard MiniAOD collection names for NanoAOD compatibility

# Vertices - standard MiniAOD name (must be produced before PF candidates)
offlineSlimmedPrimaryVertices = cms.EDProducer("Run3ScoutingVertexToRecoVertexProducer",
    src=cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx")
)

# PF Candidates - using pat::PackedCandidate for MiniAOD compatibility
# Requires vertices to be produced first for proper vertex references
packedPFCandidates = cms.EDProducer("Run3ScoutingParticleToPackedCandidateProducer",
    src=cms.InputTag("hltScoutingPFPacker"),
    vertices=cms.InputTag("offlineSlimmedPrimaryVertices"),
    tracks=cms.InputTag("scoutingTracks"),
    CHS=cms.bool(False),
    covarianceVersion=cms.int32(1),
    covarianceSchema=cms.int32(520)
)


# Muons - standard MiniAOD name
# Note: From 2024, there are two muon collections (with/without vertex)
# Default uses hltScoutingMuonPacker, 2024+ uses hltScoutingMuonPackerVtx
slimmedMuons = cms.EDProducer("PatFromScoutingMuonProducer",
    src=cms.InputTag("hltScoutingMuonPacker")
)
# For 2024 data, use hltScoutingMuonPackerVtx which includes vertex information
run3_scouting_2024.toModify(slimmedMuons, src="hltScoutingMuonPackerVtx")

# Displaced muons (no vertex constraint) - only available from 2024+
# Before 2024, there was only hltScoutingMuonPacker (no Vtx/NoVtx split)
slimmedMuonsNoVtx = cms.EDProducer("PatFromScoutingMuonProducer",
    src=cms.InputTag("hltScoutingMuonPackerNoVtx")
)

# Dimuon displaced vertices as VertexCompositePtrCandidate (like slimmedSecondaryVertices)
# with CandidatePtr daughters pointing into the corresponding muon collection.
# The muon-vertex link is derived from Run3ScoutingMuon::vtxIndx().
# Default (pre-2024): single muon collection → single dimuon vertex collection
scoutingDimuonVertices = cms.EDProducer("ScoutingDimuonVtxProducer",
    scoutingMuons=cms.InputTag("hltScoutingMuonPacker"),
    scoutingVertices=cms.InputTag("hltScoutingMuonPacker", "displacedVtx"),
    patMuons=cms.InputTag("slimmedMuons")
)
# For 2024+, Vtx muons have their own displaced vertices
run3_scouting_2024.toModify(scoutingDimuonVertices,
    scoutingMuons="hltScoutingMuonPackerVtx",
    scoutingVertices=cms.InputTag("hltScoutingMuonPackerVtx", "displacedVtx")
)

# Dimuon vertices for NoVtx muons - only available from 2024+
scoutingDimuonVerticesNoVtx = cms.EDProducer("ScoutingDimuonVtxProducer",
    scoutingMuons=cms.InputTag("hltScoutingMuonPackerNoVtx"),
    scoutingVertices=cms.InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx"),
    patMuons=cms.InputTag("slimmedMuonsNoVtx")
)

# Electrons - standard MiniAOD name
slimmedElectrons = cms.EDProducer("PatFromScoutingElectronProducer",
    src=cms.InputTag("hltScoutingEgammaPacker")
)

# Photons - standard MiniAOD name
slimmedPhotons = cms.EDProducer("PatFromScoutingPhotonProducer",
    src=cms.InputTag("hltScoutingEgammaPacker")
)

# Jets - standard MiniAOD name
slimmedJets = cms.EDProducer("PatFromScoutingJetProducer",
    src=cms.InputTag("hltScoutingPFPacker"),
    pfCandidates=cms.InputTag("packedPFCandidates")
)

# MET - standard MiniAOD name
# Uses precomputed MET from scouting data
slimmedMETs = cms.EDProducer("Run3ScoutingMETProducer",
    metPt=cms.InputTag("hltScoutingPFPacker", "pfMetPt"),
    metPhi=cms.InputTag("hltScoutingPFPacker", "pfMetPhi")
)

# Tracks
scoutingTracks = cms.EDProducer("Run3ScoutingTrackToRecoTrackProducer",
    src=cms.InputTag("hltScoutingTrackPacker")
)

# Beam spot from conditions database
offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

# Rho (pileup density) - copy from HLT scouting data
fixedGridRhoFastjetAll = cms.EDProducer("ScoutingRhoProducer",
    src=cms.InputTag("hltScoutingPFPacker", "rho")
)

# L1 trigger unpacking from raw FED data
# Scouting stores L1 raw data in hltFEDSelectorL1
gtStage2Digis = cms.EDProducer("L1TRawToDigi",
    InputLabel = cms.InputTag("hltFEDSelectorL1"),
    Setup = cms.string("stage2::GTSetup"),
    FedIds = cms.vint32(1404),
)

# L1 objects with standard module names for downstream compatibility
# Standard MiniAOD has L1 muons from gmtStage2Digis and L1 calo objects from caloStage2Digis
# These producers copy from gtStage2Digis to provide the expected module labels
gmtStage2Digis = cms.EDProducer("Run3ScoutingL1MuonProducer",
    muonSource = cms.InputTag("gtStage2Digis", "Muon")
)

caloStage2Digis = cms.EDProducer("Run3ScoutingL1CaloProducer",
    jetSource = cms.InputTag("gtStage2Digis", "Jet"),
    egammaSource = cms.InputTag("gtStage2Digis", "EGamma"),
    tauSource = cms.InputTag("gtStage2Digis", "Tau"),
    etsumSource = cms.InputTag("gtStage2Digis", "EtSum")
)

# Task containing all producers
scoutingToMiniAODTask = cms.Task(
    packedPFCandidates,
    offlineSlimmedPrimaryVertices,
    offlineBeamSpot,
    slimmedMuons,
    scoutingDimuonVertices,
    slimmedElectrons,
    slimmedPhotons,
    slimmedJets,
    slimmedMETs,
    scoutingTracks,
    fixedGridRhoFastjetAll,
    gtStage2Digis,
    gmtStage2Digis,
    caloStage2Digis
)

# For 2024+, add NoVtx muon collection and its dimuon vertices
_scoutingToMiniAODTask_2024 = scoutingToMiniAODTask.copy()
_scoutingToMiniAODTask_2024.add(slimmedMuonsNoVtx)
_scoutingToMiniAODTask_2024.add(scoutingDimuonVerticesNoVtx)
run3_scouting_2024.toReplaceWith(scoutingToMiniAODTask, _scoutingToMiniAODTask_2024)

# Sequence for backward compatibility
scoutingToMiniAODSequence = cms.Sequence(scoutingToMiniAODTask)


# ============================================================
# Customization function for cmsDriver
# ============================================================

def customiseScoutingToMiniAOD(process):
    """
    Minimal customization for scouting to MiniAOD conversion.

    Usage with cmsDriver (for 2024 data):

        cmsDriver.py scoutMini \\
            --scenario pp \\
            --conditions auto:run3_data_prompt \\
            --era Run3_2024 \\
            --eventcontent MINIAOD \\
            --datatier MINIAOD \\
            --step USER:PhysicsTools/PatFromScouting/scoutingToMiniAOD_cff.scoutingToMiniAODTask \\
            --filein file:scouting.root \\
            --fileout file:scoutingMiniAOD.root \\
            --customise PhysicsTools/PatFromScouting/scoutingToMiniAOD_cff.customiseScoutingToMiniAOD \\
            --data --no_exec -n 100

    For 2025 data, use --era Run3_2025

    The customization only:
    - Loads particle data table (needed for PackedCandidate producer)
    - Extends output to include scoutingTracks and L1 collections

    The --step USER:...Task handles adding the producers.
    The --eventcontent MINIAOD provides standard MiniAOD output commands.
    """

    # Load particle data table (needed for PackedCandidate producer)
    process.load("SimGeneral.HepPDTESSource.pdt_cfi")

    # Handle missing collections gracefully (not all scouting triggers save all objects)
    # This allows processing datasets where some events don't have egamma, etc.
    if hasattr(process, 'options'):
        if not hasattr(process.options, 'TryToContinue'):
            process.options.TryToContinue = cms.untracked.vstring()
        process.options.TryToContinue.append('ProductNotFound')
    else:
        process.options = cms.untracked.PSet(
            TryToContinue = cms.untracked.vstring('ProductNotFound')
        )

    # Extend output commands to include scouting-specific collections
    # Standard MINIAOD eventcontent already keeps slimmedMuons, slimmedJets, etc.
    # We add scoutingTracks, L1 collections, rho, and drop raw scouting collections
    for name in ['MINIAODoutput', 'MINIAODSIMoutput', 'output', 'out']:
        if hasattr(process, name):
            outputModule = getattr(process, name)
            outputModule.outputCommands.extend([
                # Keep scouting-specific collections
                'keep patMuons_slimmedMuonsNoVtx_*_*',
                'keep *_scoutingDimuonVertices_*_*',
                'keep *_scoutingDimuonVerticesNoVtx_*_*',
                'keep recoTracks_scoutingTracks__*',
                'keep *_scoutingTracks_vertexIndex_*',
                'keep *_gtStage2Digis_*_*',
                'keep *_gmtStage2Digis_*_*',
                'keep *_caloStage2Digis_*_*',
                'keep *_fixedGridRhoFastjetAll_*_*',
                # Drop raw scouting collections to save space
                'drop *_hltScouting*_*_*',
                'drop *_hltFEDSelectorL1_*_*',
            ])
            break

    return process

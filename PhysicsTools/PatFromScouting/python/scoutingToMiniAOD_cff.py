import FWCore.ParameterSet.Config as cms

# PF Candidates (using existing producer from PhysicsTools/Scouting)
scoutingPFCandidates = cms.EDProducer("Run3ScoutingParticleToRecoPFCandidateProducer",
    scoutingparticle=cms.InputTag("hltScoutingPFPacker"),
    softKiller=cms.bool(False),
    CHS=cms.bool(False)
)

# Vertices
scoutingVertices = cms.EDProducer("Run3ScoutingVertexToRecoVertexProducer",
    src=cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx")
)

# Muons
scoutingMuons = cms.EDProducer("PatFromScoutingMuonProducer",
    src=cms.InputTag("hltScoutingMuonPacker")
)

# Electrons
scoutingElectrons = cms.EDProducer("PatFromScoutingElectronProducer",
    src=cms.InputTag("hltScoutingEgammaPacker")
)

# Photons
scoutingPhotons = cms.EDProducer("PatFromScoutingPhotonProducer",
    src=cms.InputTag("hltScoutingEgammaPacker")
)

# Jets
scoutingJets = cms.EDProducer("PatFromScoutingJetProducer",
    src=cms.InputTag("hltScoutingPFPacker")
)

# MET
scoutingMET = cms.EDProducer("Run3ScoutingMETProducer",
    src=cms.InputTag("hltScoutingPFPacker")
)

# Tracks
scoutingTracks = cms.EDProducer("Run3ScoutingTrackToRecoTrackProducer",
    src=cms.InputTag("hltScoutingTrackPacker")
)

# Task containing all producers
scoutingToMiniAODTask = cms.Task(
    scoutingPFCandidates,
    scoutingVertices,
    scoutingMuons,
    scoutingElectrons,
    scoutingPhotons,
    scoutingJets,
    scoutingMET,
    scoutingTracks
)

# Sequence for backward compatibility
scoutingToMiniAODSequence = cms.Sequence(scoutingToMiniAODTask)

import FWCore.ParameterSet.Config as cms

scoutingMuonExistenceFilter = cms.EDFilter("Run3ScoutingMuonExistenceFilter",
    product = cms.InputTag("hltScoutingMuonPacker")
)

scoutingMuonDisplacedVertexExistenceFilter = cms.EDFilter("Run3ScoutingVertexExistenceFilter",
    product = cms.InputTag("hltScoutingMuonPacker", "displacedVtx")
)

scoutingMuonNoVtxExistenceFilter = cms.EDFilter("Run3ScoutingMuonExistenceFilter",
    product = cms.InputTag("hltScoutingMuonPackerNoVtx")
)

scoutingMuonNoVtxDisplacedVertexExistenceFilter = cms.EDFilter("Run3ScoutingVertexExistenceFilter",
    product = cms.InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx")
)

scoutingMuonVtxExistenceFilter = cms.EDFilter("Run3ScoutingMuonExistenceFilter",
    product = cms.InputTag("hltScoutingMuonPackerVtx")
)

scoutingMuonVtxDisplacedVertexExistenceFilter = cms.EDFilter("Run3ScoutingVertexExistenceFilter",
    product = cms.InputTag("hltScoutingMuonPackerVtx", "displacedVtx")
)

scoutingElectronExistenceFilter = cms.EDFilter("Run3ScoutingElectronExistenceFilter",
    product = cms.InputTag("hltScoutingEgammaPacker")
)

scoutingPhotonExistenceFilter = cms.EDFilter("Run3ScoutingPhotonExistenceFilter",
    product = cms.InputTag("hltScoutingEgammaPacker")
)

scoutingTrackExistenceFilter = cms.EDFilter("Run3ScoutingTrackExistenceFilter",
    product = cms.InputTag("hltScoutingTrackPacker")
)

scoutingPrimaryVertexExistenceFilter = cms.EDFilter("Run3ScoutingVertexExistenceFilter",
    product = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx")
)

scoutingParticleExistenceFilter = cms.EDFilter("Run3ScoutingParticleExistenceFilter",
    product = cms.InputTag("hltScoutingPFPacker")
)

scoutingPFJetExistenceFilter = cms.EDFilter("Run3ScoutingPFJetExistenceFilter",
    product = cms.InputTag("hltScoutingPFPacker")
)

scoutingMETptExistenceFilter = cms.EDFilter("DoubleExistenceFilter",
    product = cms.InputTag("hltScoutingPFPacker", "pfMetPt")
)

scoutingMETphiExistenceFilter = cms.EDFilter("DoubleExistenceFilter",
    product = cms.InputTag("hltScoutingPFPacker", "pfMetPhi")
)

scoutingRhoExistenceFilter = cms.EDFilter("DoubleExistenceFilter",
    product = cms.InputTag("hltScoutingPFPacker", "rho")
)

# from 2024, there are two scouting muon collections (https://its.cern.ch/jira/browse/CMSHLT-3089)
from Configuration.Eras.Modifier_run3_scouting_nanoAOD_2024_cff import run3_scouting_nanoAOD_2024
scoutingMuonExistenceFilterSequence = cms.Sequence(scoutingMuonExistenceFilter*scoutingMuonDisplacedVertexExistenceFilter)
run3_scouting_nanoAOD_2024.toReplaceWith(scoutingMuonExistenceFilterSequence,
    cms.Sequence(scoutingMuonNoVtxExistenceFilter*scoutingMuonNoVtxDisplacedVertexExistenceFilter\
                *scoutingMuonVtxExistenceFilter*scoutingMuonVtxDisplacedVertexExistenceFilter)
)

scoutingExistenceFilter = cms.Sequence(
    scoutingMuonExistenceFilterSequence
    *scoutingElectronExistenceFilter
    *scoutingPhotonExistenceFilter
    *scoutingTrackExistenceFilter
    *scoutingPrimaryVertexExistenceFilter
    *scoutingParticleExistenceFilter
    *scoutingPFJetExistenceFilter
    *scoutingMETptExistenceFilter
    *scoutingMETphiExistenceFilter
    *scoutingRhoExistenceFilter
)

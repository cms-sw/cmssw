import FWCore.ParameterSet.Config as cms

zToMuMuGoldenEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCandidatesOwned_allMuons_*_*', 'keep *_allMuonIsolations_*_*', 'keep *_zToMuMuGolden_*_*', 'keep *_allMuonsGenParticlesMatch_*_*', 'keep *_zToMuMuGoldenGenParticlesMatch_*_*')
)
zToMuMuGoldenEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToMuMuGoldenHLTPath')
    )
)


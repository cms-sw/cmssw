import FWCore.ParameterSet.Config as cms

tauTo3MuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCandidatesOwned_genParticle_*_*', 
        'keep *_tauTo3MuFilter_*_*')
)
tauTo3MuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('tauTo3MuPath')
    )
)


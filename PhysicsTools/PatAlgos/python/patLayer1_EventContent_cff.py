# The following comments couldn't be translated into the new config version:

# to be removed when full GenParticle transition finished
# to be removed when no external MC ref's left
import FWCore.ParameterSet.Config as cms

patLayer1EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_TriggerResults_*_*', 
        'keep *_genParticleCandidates_*_*', 
        'keep *_genParticles_*_*', 
        'keep recoJetTags_*_*_*', 
        'drop recoJetTags_coneIsolationTauJetTags_*_*', 
        'keep recoTracks_gsWithMaterialTracks_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep *_selectedLayer1Photons_*_*', 
        'keep *_selectedLayer1Electrons_*_*', 
        'keep *_selectedLayer1Muons_*_*', 
        'keep *_selectedLayer1Taus_*_*', 
        'keep *_selectedLayer1Jets_*_*', 
        'keep *_selectedLayer1METs_*_*', 
        'keep *_selectedLayer1Hemispheres_*_*')
)


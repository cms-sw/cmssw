# The following comments couldn't be translated into the new config version:

# to be removed when no external MC ref's left
#"keep recoTrackIPTagInfos_*_*_*",         // no longer needed (embedded by value)
#"keep recoSoftLeptonTagInfos_*_*_*",  
#"keep recoSecondaryVertexTagInfos_*_*_*",

import FWCore.ParameterSet.Config as cms

patLayer1EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_TriggerResults_*_*', 
        'keep *_genParticles_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep *_selectedLayer1Photons_*_*', 
        'keep *_selectedLayer1Electrons_*_*', 
        'keep *_selectedLayer1Muons_*_*', 
        'keep *_selectedLayer1Taus_*_*', 
        'keep *_selectedLayer1Jets_*_*', 
        'keep *_selectedLayer1METs_*_*', 
        'keep *_selectedLayer1Hemispheres_*_*')
)


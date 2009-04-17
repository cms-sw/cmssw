# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

patLayer1EventContent = cms.PSet(
  outputCommands = cms.untracked.vstring(
    'keep recoGenParticles_genParticles_*_*',
    'keep *_genEventScale_*_*',
    'keep *_genEventWeight_*_*',
    'keep *_genEventPdfInfo_*_*',
    'keep edmTriggerResults_TriggerResults_*_HLT', 
    'keep *_hltTriggerSummaryAOD_*_*',
    'keep *_offlineBeamSpot_*_*',
    'keep *_offlinePrimaryVertices_*_*',
    'keep recoTracks_generalTracks_*_*', 
    'keep *_towerMaker_*_*',
    'keep *_selectedLayer1Photons_*_*', 
    'keep *_selectedLayer1Electrons_*_*', 
    'keep *_selectedLayer1Muons_*_*', 
    'keep *_selectedLayer1Taus_*_*', 
    'keep *_selectedLayer1Jets_*_*', 
    'keep *_selectedLayer1METs_*_*',
    'keep patPFParticles_*_*_*',
    'keep *_selectedLayer1Hemispheres_*_*'
  )
)


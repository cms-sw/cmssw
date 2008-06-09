import FWCore.ParameterSet.Config as cms

JetTagLifetimeHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*')
)


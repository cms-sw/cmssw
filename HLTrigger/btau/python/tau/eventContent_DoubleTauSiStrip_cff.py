import FWCore.ParameterSet.Config as cms

DoubleTauSiStripHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1_*_*', 
        'keep *_hltIcone5Tau2_*_*', 
        'keep *_hltIcone5Tau3_*_*', 
        'keep *_hltIcone5Tau4_*_*', 
        'keep *_hltL2TauJetsProvider_*_*', 
        'keep *_hltEcalSingleTauIsolated_*_*', 
        'keep *_hltEcalDoubleTauIsolated_*_*', 
        'keep *_hltMet_*_*', 
        'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL25DoubleTau_*_*', 
        'keep *_hltAssociatorL25SingleTau_*_*', 
        'keep *_hltAssociatorL25DoubleTau_*_*', 
        'keep *_hltConeIsolationL25SingleTau_*_*', 
        'keep *_hltConeIsolationL25DoubleTau_*_*', 
        'keep *_hltIsolatedL25SingleTau_*_*', 
        'keep *_hltIsolatedL25DoubleTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL3DoubleTau_*_*', 
        'keep *_hltAssociatorL3DoubleTau_*_*', 
        'keep *_hltConeIsolationL3DoubleTau_*_*', 
        'keep *_hltIsolatedL3DoubleTau_*_*')
)


import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
fouStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
fouStep.src = 'iterativeFourthTrackMerging'
fouStep.keepAllTracks = True
fouStep.copyExtras = True
fouStep.copyTrajectories = True
fouStep.chi2n_par = 0.3
fouStep.res_par = ( 0.003, 0.001 )
fouStep.minNumberLayers = 5
fouStep.d0_par1 = ( 1.0, 4.0 )
fouStep.dz_par1 = ( 1.0, 4.0 )
fouStep.d0_par2 = ( 1.0, 4.0 )
fouStep.dz_par2 = ( 1.0, 4.0 )
iterativeFourthTrackFiltering = cms.Sequence(fouStep)



import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
fifthStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
fifthStep.src = 'iterativeFifthTrackMerging'
fifthStep.keepAllTracks = True
fifthStep.copyExtras = True
fifthStep.copyTrajectories = True
fifthStep.chi2n_par = 0.25
fifthStep.res_par = ( 0.003, 0.001 )
fifthStep.minNumberLayers = 6
fifthStep.d0_par1 = ( 1.2, 4.0 )
fifthStep.dz_par1 = ( 1.1, 4.0 )
fifthStep.d0_par2 = ( 1.2, 4.0 )
fifthStep.dz_par2 = ( 1.1, 4.0 )
iterativeFifthTrackFiltering = cms.Sequence(fifthStep)



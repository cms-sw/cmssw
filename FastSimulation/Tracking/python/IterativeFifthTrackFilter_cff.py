import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
fifthStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeFifthTrackMerging',
##keepAllTracks = True,
copyExtras = True,
copyTrajectories = True,
chi2n_par = 0.25,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 6,
minNumber3DLayers = 2,
maxNumberLostLayers = 0,
d0_par1 = ( 1.2, 4.0 ),
dz_par1 = ( 1.1, 4.0 ),
d0_par2 = ( 1.2, 4.0 ),
dz_par2 = ( 1.1, 4.0 )
)
iterativeFifthTrackFiltering = cms.Sequence(fifthStep)



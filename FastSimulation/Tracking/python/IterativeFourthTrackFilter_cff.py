import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
fouStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeFourthTrackMerging',
##keepAllTracks = True
copyExtras = True,
copyTrajectories = True,
chi2n_par = 0.3,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 6,
minNumber3DLayers = 3,
maxNumberLostLayers = 0,
d0_par1 = ( 1.0, 4.0 ),
dz_par1 = ( 1.0, 4.0 ),
d0_par2 = ( 1.0, 4.0 ),
dz_par2 = ( 1.0, 4.0 )
)

iterativeFourthTrackFiltering = cms.Sequence(fouStep)



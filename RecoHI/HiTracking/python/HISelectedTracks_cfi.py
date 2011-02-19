import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.selectHighPurity_cfi import selectHighPurity

#loose
hiTracksWithLooseQuality = selectHighPurity.clone(
    src = "hiGlobalPrimTracks",
    keepAllTracks = False,
    
    vertices = "hiSelectedVertex",
    vertexCut = "ndof>=0&((chi2==0.0)|(chi2prob(chi2,ndof)>=0.01))",
    copyTrajectories = True,             # needed by TrackClusterRemover
    
    qualityBit = 'loose',                # set to '' if you don't want to set the bit

    chi2n_par = 1.6,                     # normalizedChi2 < nLayers * chi2n_par
    res_par = [0.003, 0.001],            # residual parameterization (re-check in HI)
    d0_par1 = [9999, 1],                 # parameterized nomd0E
    dz_par1 = [9999, 1],
    d0_par2 = [300.0, 0.3],              # d0E from tk.d0Error
    dz_par2 = [300.0, 0.3],

    minNumberLayers = 3,
    minNumber3DLayers = 3,
    maxNumberLostLayers = 999
)

#tight
hiTracksWithTightQuality = hiTracksWithLooseQuality.clone(
    src = "hiTracksWithLooseQuality",
    keepAllTracks = False,
    qualityBit = 'tight',
    chi2n_par = 0.7,   
    d0_par2 = [30.0, 0.3],
    dz_par2 = [30.0, 0.3],
    minNumberLayers = 5,
)

#highPurity
hiSelectedTracks = hiTracksWithLooseQuality.clone(
    src = "hiTracksWithTightQuality",
    keepAllTracks = False,
    qualityBit = 'highPurity',
    chi2n_par = 0.4,    
    d0_par2 = [5.0, 0.3],               
    dz_par2 = [30.0, 0.3],
    minNumberLayers = 7,
)

#complete sequence
hiTracksWithQuality = cms.Sequence(hiTracksWithLooseQuality
                                   * hiTracksWithTightQuality
                                   * hiSelectedTracks)

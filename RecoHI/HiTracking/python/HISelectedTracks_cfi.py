import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.selectHighPurity_cfi import selectHighPurity

#loose
hiTracksWithLooseQuality = selectHighPurity.clone(
    src = "hiGlobalPrimTracks",
    keepAllTracks = False,
    # Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts = cms.bool(True),    
    vertices = "hiSelectedVertex",
    vertexCut = "",    
    copyTrajectories = True,             # needed by TrackClusterRemover

    useVertices = True,
    useHICuts = True,    

    qualityBit = 'loose',                # set to '' if you don't want to set the bit
    
    min_nhits = cms.int32(10),
    max_relpterr = 0.1,
    chi2n_par = 0.2,                     # normalizedChi2 < nLayers * chi2n_par
    d0_par2 = [8.0, 0.0],              # d0E from tk.d0Error
    dz_par2 = [8.0, 0.0],
    
    
    d0_par1 = [9999., 0.],                 # parameterized nomd0E
    dz_par1 = [9999., 0.],
    res_par = [99999., 99999.],            # residual parameterization (re-check in HI)
    nSigmaZ = 9999.,

    max_z0 = cms.double(1000),
    max_d0 = cms.double(1000),

    minNumberLayers = 0,
    minNumber3DLayers = 0,
    maxNumberLostLayers = 999
    )

#tight
hiTracksWithTightQuality = hiTracksWithLooseQuality.clone(
    src = "hiGlobalPrimTracks",
    keepAllTracks = False,
    qualityBit = 'tight',
    min_nhits = cms.int32(12),
    max_relpterr = 0.05,
    chi2n_par = 0.15,
    d0_par2 = [5.0, 0.0],
    dz_par2 = [5.0, 0.0]
    )

#highPurity
hiSelectedTracks = hiTracksWithLooseQuality.clone(
    src = "hiGlobalPrimTracks",
    keepAllTracks = False,
    qualityBit = 'highPurity',
    min_nhits = cms.int32(13),
    max_relpterr = 0.05,
    chi2n_par = 0.15,
    d0_par2 = [3.0, 0.0],
    dz_par2 = [3.0, 0.0]
    )

#complete sequence
hiTracksWithQuality = cms.Sequence(hiTracksWithLooseQuality
                                   * hiTracksWithTightQuality
                                   * hiSelectedTracks)

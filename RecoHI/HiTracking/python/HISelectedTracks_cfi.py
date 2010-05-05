import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.selectHighPurity_cfi import selectHighPurity

hiSelectedTracks = selectHighPurity.clone(
    
    src = cms.InputTag("hiGlobalPrimTracks"),
    
    vertices = cms.InputTag("hiSelectedVertex"),
    vertexCut = cms.string('ndof>=0&((chi2==0.0)|(chi2prob(chi2,ndof)>=0.01))'),
    copyTrajectories = cms.untracked.bool(True),  # needed by TrackClusterRemover
    
    qualityBit = cms.string(''),         # set to '' if you don't want to set the bit

    chi2n_par = cms.double(0.4),         # normalizedChi2 < nLayers * chi2n_par
    res_par = cms.vdouble(0.003, 0.001), # residual parameterization (re-check in HI)
    d0_par1 = cms.vdouble(9999, 1),      # parameterized nomd0E
    dz_par1 = cms.vdouble(9999, 1),
    d0_par2 = cms.vdouble(5.0, 0.3),     # d0E from tk.d0Error
    dz_par2 = cms.vdouble(30.0, 0.3),

    minNumberLayers = cms.uint32(7),
    minNumber3DLayers = cms.uint32(3),
    maxNumberLostLayers = cms.uint32(999)
    
    )

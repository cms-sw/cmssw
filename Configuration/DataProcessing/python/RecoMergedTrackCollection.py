import FWCore.ParameterSet.Config as cms

def customiseAddMergedTrackCollection(process):

    ####Customize process to add retuned Merged Track Collection in the reco sequence 

    ###Set the new parameters for pixel tracks, general tracks and
    ###merging procedure
    
    ##Use clusterShapeFilter for pixel tracks reconstruction
    process.hiConformalPixelTracks.FilterPSet.useClusterShape = cms.bool(True)
    ##relax chi2 cuts in pixel tracks
    process.hiPixelOnlyStepSelector.trackSelectors[0].chi2n_no1Dmod_par = cms.double(9999.9)
    process.hiPixelOnlyStepSelector.trackSelectors[1].chi2n_no1Dmod_par = cms.double(9999.9)
    process.hiPixelOnlyStepSelector.trackSelectors[2].chi2n_no1Dmod_par = cms.double(18.0)
    process.hiPixelOnlyStepSelector.trackSelectors[0].chi2n_par = cms.double(9999.9)
    process.hiPixelOnlyStepSelector.trackSelectors[1].chi2n_par = cms.double(9999.9)
    process.hiPixelOnlyStepSelector.trackSelectors[2].chi2n_par = cms.double(9999.9)
    ##Only consider the last pset to select the pixel tracks before merging with hiGeneralTracks
    process.hiPixelOnlyStepSelector.trackSelectors[0].keepAllTracks = cms.bool(True)
    process.hiPixelOnlyStepSelector.trackSelectors[1].keepAllTracks = cms.bool(True)
    process.hiPixelOnlyStepSelector.trackSelectors[2].keepAllTracks = cms.bool(False)
    ##Retune parameters for merging - All are default except "min_nhits" selection in hiGeneralTracks
    process.hiHighPtStepSelector.trackSelectors[0].min_nhits = cms.uint32(0)
    #pt cuts
    process.hiHighPtStepSelector.trackSelectors[0].pixel_pTMinCut[0] = 1.0
    process.hiHighPtStepSelector.trackSelectors[0].pixel_pTMinCut[1] = 1.8
    process.hiPixelOnlyStepSelector.trackSelectors[2].pixel_pTMaxCut[1] = 1.6
    process.hiPixelOnlyStepSelector.trackSelectors[2].pixel_pTMaxCut[0] = 2.4
    ##alpha
    process.hiHighPtStepSelector.trackSelectors[0].pixel_pTMinCut[3] = 2.5
    process.hiPixelOnlyStepSelector.trackSelectors[2].pixel_pTMaxCut[3] = 2.5
    ##y
    process.hiHighPtStepSelector.trackSelectors[0].pixel_pTMinCut[2] = 0.15
    process.hiPixelOnlyStepSelector.trackSelectors[2].pixel_pTMaxCut[2] = 12

    ###keep only merged collection
    #process.AODoutput.outputCommands += ['keep *_hiConformalPixelTracks_*_*']
    process.AODoutput.outputCommands += ['drop *_hiGeneralTracks_*_*']

    ###Add pixel tracks and merging procedure in the sequence
    process.reconstruction_step += process.hiMergedConformalPixelTracking
    

    return process

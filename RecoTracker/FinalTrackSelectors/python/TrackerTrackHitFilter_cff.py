import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.trackerTrackHitFilter_cfi import trackerTrackHitFilter as _trackerTrackHitFilter
TrackerTrackHitFilter = _trackerTrackHitFilter.clone(
    src = "generalTracks",
    minimumHits = 3, ##min number of hits for refit
    ## # layers to remove
    commands = ["drop PXB",  "drop PXE"],   ### same works for TIB, TID, TOB, TEC,
                                            # "drop TIB 3",  ## you can also drop specific layers/wheel/disks
                                            # "keep PXB 3",  ## you can also 'keep' some layer after
                                            # having dropped the whole structure

    ###list of individual detids to turn off, in addition to the structures above
    detsToIgnore = [],

    ### what to do with invalid hits
    replaceWithInactiveHits = False, ## instead of removing hits replace
                                     ## them with inactive hits, so you still
                                     ## consider the multiple scattering

    stripFrontInvalidHits = False,   ## strip invalid & inactive hits from
    stripBackInvalidHits = False,    ## any end of the track

    stripAllInvalidHits = False, ## not sure if it's better 'true' or 'false'
                                 ## might be dangerous to turn on
                                 ## as you will forget about MS

    ### hit quality cuts
    isPhase2 = False,
    rejectBadStoNHits = False,
    CMNSubtractionMode = "Median", ## "TT6"
    StoNcommands = ["TIB 1.0 ", "TOB 1.0 999.0"],
    useTrajectories = False,
    rejectLowAngleHits = False,
    TrackAngleCut = 0.25,       ## in radians
    tagOverlaps= False,
    usePixelQualityFlag = False,
    PxlTemplateProbXYCut = 0.000125,   # recommended by experts
    PxlTemplateProbXYChargeCut = -99., # recommended by experts
    PxlTemplateqBinCut = [0, 3],       # recommended by experts
    PxlCorrClusterChargeCut = -999.0
) #### end of module

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(TrackerTrackHitFilter,
                        isPhase2 = True)

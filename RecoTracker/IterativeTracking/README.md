Iterative tracking
==================

Even if we have some automation for propagating the tracking
iterations and their order to various places, there are still certain
steps that need to be followed. See below for instructions.


How to add a totally new tracking iteration?
--------------------------------------------

1. If you are not using an existing track algo enumerator, add one (or
   modify one of the `reservedForUpgrades*`) to
   * [DataFormats/TrackReco/interface/TrackBase.h](../../DataFormats/TrackReco/interface/TrackBase.h)
   * [RecoTracker/FinalTrackSelectors/src/trackAlgoPriorityOrder.h](../../RecoTracker/FinalTrackSelectors/src/trackAlgoPriorityOrder.h)
2. If the algo is something not used previously in offline tracking, add it to
   * DQM: [DQM/TrackingMonitorSource/python/IterTrackingModules4seedMonitoring_cfi.py](../../DQM/TrackingMonitorSource/python/IterTrackingModules4seedMonitoring_cfi.py)
   * Validation plot scripts: [Validation/RecoTrack/python/plotting/trackingPlots.py](../../Validation/RecoTrack/python/plotting/trackingPlots.py)
3. Add a `<iteration>_cff.py` file to RecoTracker/IterativeTracking/python
   * `<iteration>` must be the same as the algo enumerator, except starting with an upper case letter
   * The `cms.Sequence` defined in the file must also be named as `<iteration>`
   * All modules in the file must start with the algo enumerator name (now starting with lower case letter), also please follow the naming convention of the various types of modules
4. Import the `<iteration>_cff.py` from [RecoTracker/IterativeTracking/python/iterativeTk_cff.py](python/iterativeTk_cff.py)
5. Add the iteration to the intended tracking flavour in [RecoTracker/IterativeTracking/python/iterativeTkConfig.py](python/iterativeTkConfig.py)
6. Add the iteration to `earlyGeneralTracks` in [RecoTracker/FinalTrackSelectors/python/earlyGeneralTracks_cfi.py](../../RecoTracker/FinalTrackSelectors/python/earlyGeneralTracks_cfi.py)
   * Or if you're dealing with muon iterations, edit `preDuplicateMergingGeneralTracks` in [RecoTracker/FinalTrackSelectors/python/preDuplicateMergingGeneralTracks_cfi.py](../../RecoTracker/FinalTrackSelectors/python/preDuplicateMergingGeneralTracks_cfi.py)


How to add an existing tracking iteration to a tracking flavour where it does not exist currently?
--------------------------------------------------------------------------------------------------

Follow steps 5 and 6 above.


How to reorder iterations for a tracking flavour?
-------------------------------------------------

The iteration order for each tracking flavour is specified in
[RecoTracker/IterativeTracking/python/iterativeTkConfig.py](../../RecoTracker/IterativeTracking/python/iterativeTkConfig.py).
You need to edit that file, and possibly double-check the
`earlyGeneralTracks` configuration in
[RecoTracker/FinalTrackSelectors/python/earlyGeneralTracks_cfi.py](../../RecoTracker/FinalTrackSelectors/python/earlyGeneralTracks_cfi.py).


Tricks to temporarily and quickly disable an iteration
------------------------------------------------------

These instructions are supposed to disable an iteration in the tracking recontruction without removing it completely, i.e. the iteration will still be processed but it will not find any tracks.
1. Search in the `TrajectoryBuilder` of the specific iteration that you want to remove.
2. Search the module producing the `TrajectoryFilter` and add in the RECO configuration file the following: `process.***TrajectoryFilter.minimumNumberOfHits = 10000` where `***` is most luckly the name of the iteration.

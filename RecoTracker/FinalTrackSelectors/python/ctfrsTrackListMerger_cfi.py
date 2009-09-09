import FWCore.ParameterSet.Config as cms

#
# ctf tracks parameter-set entries for module
#
# SimpleTrackListMerger
#
# located in
#
# RecoTracker/FinalTrackSelectors
#
# 
# cleans and merges ctf and rs Track lists and put new list back in Event

import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
ctfrsTrackListMerger = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer2 = cms.string('rsWithMaterialTracks'),
    TrackProducer1 = cms.string('generalTracks')
)

print "one should use RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi instead of RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.py"



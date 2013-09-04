import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
generalTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('initialStepTracks'),),
#    hasSelector=cms.vint32(1,1,1,1,1,1,1),
    hasSelector=cms.vint32(1),    
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

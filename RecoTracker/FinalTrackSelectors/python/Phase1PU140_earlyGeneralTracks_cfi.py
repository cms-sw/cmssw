import FWCore.ParameterSet.Config as cms
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
earlyGeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('initialStepTracks'),
                      cms.InputTag('highPtTripletStepTracks'),
                      cms.InputTag('lowPtQuadStepTracks'),
                      cms.InputTag('lowPtTripletStepTracks'),
                      cms.InputTag('detachedQuadStepTracks'),
                      cms.InputTag('pixelPairStepTracks')),
    hasSelector=cms.vint32(1,1,1,1,1,1),
    indivShareFrac=cms.vdouble(1.0,0.16,0.095,0.09,0.09,0.09),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                       cms.InputTag("highPtTripletStepSelector","highPtTripletStep"),
                                       cms.InputTag("lowPtQuadStepSelector","lowPtQuadStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("detachedQuadStep"),
                                       cms.InputTag("pixelPairStepSelector","pixelPairStep")
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
)

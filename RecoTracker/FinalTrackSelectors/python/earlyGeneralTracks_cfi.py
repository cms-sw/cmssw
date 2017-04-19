import FWCore.ParameterSet.Config as cms
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
earlyGeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('initialStepTracks'),
                      cms.InputTag('jetCoreRegionalStepTracks'),
                      cms.InputTag('lowPtTripletStepTracks'),
                      cms.InputTag('pixelPairStepTracks'),
                      cms.InputTag('detachedTripletStepTracks'),
                      cms.InputTag('mixedTripletStepTracks'),
                      cms.InputTag('pixelLessStepTracks'),
                      cms.InputTag('tobTecStepTracks')),
    hasSelector=cms.vint32(1,1,1,1,1,1,1,1),
    indivShareFrac=cms.vdouble(1.0,0.19,0.16,0.19,0.13,0.11,0.11,0.09),#using 0.19 for jetCoreRegionalStep?
    priorities=cms.vdouble(1,0.5,1,1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStep"),
                                       cms.InputTag("jetCoreRegionalStepSelector","jetCoreRegionalStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                       cms.InputTag("detachedTripletStep"),
                                       cms.InputTag("mixedTripletStep"),
                                       cms.InputTag("pixelLessStep"),
                                       cms.InputTag("tobTecStepSelector","tobTecStep")
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
)

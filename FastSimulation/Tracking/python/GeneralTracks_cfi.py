import FWCore.ParameterSet.Config as cms

#generalTracks = cms.EDProducer("FastTrackMerger",
#    # new quality setting
#    newQuality = cms.untracked.string('confirmed'),
#    # set new quality for confirmed tracks
#    promoteTrackQuality = cms.untracked.bool(True),
#    TrackProducers = cms.VInputTag(
#       cms.InputTag("zeroStepFilter"),
#       cms.InputTag("zerofivefilter"),
#       cms.InputTag("firstfilter"),
#       cms.InputTag("secfilter"),
#       cms.InputTag("thfilter"),
#       cms.InputTag("foufilter"),
#       cms.InputTag("fifthfilter"),
#    )
#)

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
generalTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('initialStepTracks'),
                      cms.InputTag('lowPtTripletStepTracks'),
                      cms.InputTag('pixelPairStepTracks'),
                      cms.InputTag('detachedTripletStepTracks'),
                      cms.InputTag('mixedTripletStepTracks'),
                      cms.InputTag('pixelLessStepTracks'),
                      cms.InputTag('tobTecStepTracks'),
#### not validated yet                      cms.InputTag('muonSeededTracksOutIn'),
#### not validated yet                      cms.InputTag('muonSeededTracksInOut')
                      ),
###    hasSelector=cms.vint32(1,1,1,1,1,1,1,1,1),
    hasSelector=cms.vint32(1,1,1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                       cms.InputTag("detachedTripletStep"),
                                       cms.InputTag("mixedTripletStep"),
                                       cms.InputTag("pixelLessStepSelector","pixelLessStep"),
                                       cms.InputTag("tobTecStepSelector","tobTecStep"),
#### not validated yet                                       cms.InputTag("muonSeededTracksOutInSelector","muonSeededTracksOutInHighPurity"),
#### not validated yet                                       cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity")
                                       ),
###    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7,8), pQual=cms.bool(True) )
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

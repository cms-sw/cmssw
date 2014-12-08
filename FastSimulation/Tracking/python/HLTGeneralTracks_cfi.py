import FWCore.ParameterSet.Config as cms

#HLTgeneralTracks = cms.EDProducer("FastTrackMerger",
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
HLTgeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone( #temporary; substitute with just a clone!
    TrackProducers = (cms.InputTag('iterativeInitialTracks'),
                      cms.InputTag('iterativeLowPtTripletTracksWithTriplets'),
                      cms.InputTag('iterativePixelPairTracks'),
                      cms.InputTag('iterativeDetachedTripletTracks'),
                      cms.InputTag('iterativeMixedTripletStepTracks'),
                      cms.InputTag('iterativePixelLessTracks'),
                      cms.InputTag('iterativeTobTecTracks')),
    hasSelector=cms.vint32(1,1,1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                       cms.InputTag("detachedTripletStep"),
                                       cms.InputTag("mixedTripletStep"),
                                       cms.InputTag("pixelLessStepSelector","pixelLessStep"),
                                       cms.InputTag("tobTecStepSelector","tobTecStep")
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

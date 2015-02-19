import FWCore.ParameterSet.Config as cms

trackClusterRemover = cms.EDProducer("TrackClusterRemover",
    maxChi2                                  = cms.double(30.0),
    trajectories                             = cms.InputTag("tracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag(""),
    overrideTrkQuals                         = cms.InputTag(''),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0), # this is used only if TrackQuality exists
                                                             # if it is not available, its value is set to 0
)

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

from RecoTracker.FinalTrackSelectors.SingleLongTrackProducer_cfi import *

from RecoTracker.FinalTrackSelectors.trackerTrackHitFilter_cfi import trackerTrackHitFilter as _trackerTrackHitFilter
TrackerTrackHitFilter = _trackerTrackHitFilter.clone(src = "SingleLongTrackProducer",
                                                     truncateTracks = True,
                                                     replaceWithInactiveHits = True,
                                                     rejectBadStoNHits = True,
                                                     usePixelQualityFlag = True)

TrackerTrackHitFilter3 = TrackerTrackHitFilter.clone(minimumHits = 3,
                                                     layersRemaining = 3)

TrackerTrackHitFilter4 = TrackerTrackHitFilter.clone(minimumHits = 4,
                                                     layersRemaining = 4)

TrackerTrackHitFilter5 = TrackerTrackHitFilter.clone(minimumHits = 5,
                                                     layersRemaining = 5)

TrackerTrackHitFilter6 = TrackerTrackHitFilter.clone(minimumHits = 6,
                                                     layersRemaining = 6)

TrackerTrackHitFilter7 = TrackerTrackHitFilter.clone(minimumHits = 7,
                                                     layersRemaining = 7)

TrackerTrackHitFilter8 = TrackerTrackHitFilter.clone(minimumHits = 8,
                                                     layersRemaining = 8)

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff
HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(src = 'TrackerTrackHitFilter')

HitFilteredTracks3 = HitFilteredTracks.clone(src = 'TrackerTrackHitFilter3')
HitFilteredTracks4 = HitFilteredTracks.clone(src = 'TrackerTrackHitFilter4')
HitFilteredTracks5 = HitFilteredTracks.clone(src = 'TrackerTrackHitFilter5')
HitFilteredTracks6 = HitFilteredTracks.clone(src = 'TrackerTrackHitFilter6')
HitFilteredTracks7 = HitFilteredTracks.clone(src = 'TrackerTrackHitFilter7')
HitFilteredTracks8 = HitFilteredTracks.clone(src = 'TrackerTrackHitFilter8')

from DQM.TrackingMonitorSource.shortenedTrackResolution_cfi import shortenedTrackResolution as _shortenedTrackResolution
trackingResolution = _shortenedTrackResolution.clone(folderName           = "Tracking/ShortTrackResolution",
                                                     hitsRemainInput      = ["3","4","5","6","7","8"],
                                                     minTracksEtaInput    = 0.0,
                                                     maxTracksEtaInput    = 2.2,
                                                     minTracksPtInput     = 15.0,
                                                     maxTracksPtInput     = 99999.9,
                                                     maxDrInput           = 0.01,
                                                     tracksInputTag       = "SingleLongTrackProducer",
                                                     tracksRerecoInputTag = ["HitFilteredTracks3","HitFilteredTracks4","HitFilteredTracks5","HitFilteredTracks6","HitFilteredTracks7","HitFilteredTracks8"])
                                                     
shortTrackResolution3to8 = cms.Sequence(SingleLongTrackProducer *
                                        TrackerTrackHitFilter3 *
                                        TrackerTrackHitFilter4 *
                                        TrackerTrackHitFilter5 *
                                        TrackerTrackHitFilter6 *
                                        TrackerTrackHitFilter7 *
                                        TrackerTrackHitFilter8 *
                                        HitFilteredTracks3 *
                                        HitFilteredTracks4 *
                                        HitFilteredTracks5 *
                                        HitFilteredTracks6 *
                                        HitFilteredTracks7 *
                                        HitFilteredTracks8 *
                                        trackingResolution)

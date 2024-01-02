import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

from RecoTracker.FinalTrackSelectors.SingleLongTrackProducer_cfi import *

from RecoTracker.FinalTrackSelectors.trackerTrackHitFilter_cfi import trackerTrackHitFilter as _trackerTrackHitFilter
ShortTrackCandidates = _trackerTrackHitFilter.clone(src = "SingleLongTrackProducer",
                                                    truncateTracks = True,
                                                    replaceWithInactiveHits = True,
                                                    rejectBadStoNHits = True,
                                                    usePixelQualityFlag = True)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(ShortTrackCandidates,
                        isPhase2 = True)

ShortTrackCandidates3 = ShortTrackCandidates.clone(minimumHits = 3,
                                                   layersRemaining = 3)

ShortTrackCandidates4 = ShortTrackCandidates.clone(minimumHits = 4,
                                                   layersRemaining = 4)

ShortTrackCandidates5 = ShortTrackCandidates.clone(minimumHits = 5,
                                                   layersRemaining = 5)

ShortTrackCandidates6 = ShortTrackCandidates.clone(minimumHits = 6,
                                                   layersRemaining = 6)

ShortTrackCandidates7 = ShortTrackCandidates.clone(minimumHits = 7,
                                                   layersRemaining = 7)

ShortTrackCandidates8 = ShortTrackCandidates.clone(minimumHits = 8,
                                                   layersRemaining = 8)

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff
RefittedShortTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(src = 'ShortTrackCandidates')

RefittedShortTracks3 = RefittedShortTracks.clone(src = 'ShortTrackCandidates3')
RefittedShortTracks4 = RefittedShortTracks.clone(src = 'ShortTrackCandidates4')
RefittedShortTracks5 = RefittedShortTracks.clone(src = 'ShortTrackCandidates5')
RefittedShortTracks6 = RefittedShortTracks.clone(src = 'ShortTrackCandidates6')
RefittedShortTracks7 = RefittedShortTracks.clone(src = 'ShortTrackCandidates7')
RefittedShortTracks8 = RefittedShortTracks.clone(src = 'ShortTrackCandidates8')

from DQM.TrackingMonitorSource.shortenedTrackResolution_cfi import shortenedTrackResolution as _shortenedTrackResolution
trackingResolution = _shortenedTrackResolution.clone(folderName           = "Tracking/ShortTrackResolution",
                                                     hitsRemainInput      = ["3","4","5","6","7","8"],
                                                     minTracksEtaInput    = 0.0,
                                                     maxTracksEtaInput    = 2.2,
                                                     minTracksPtInput     = 15.0,
                                                     maxTracksPtInput     = 99999.9,
                                                     maxDrInput           = 0.01,
                                                     tracksInputTag       = "SingleLongTrackProducer",
                                                     tracksRerecoInputTag = ["RefittedShortTracks3",
                                                                             "RefittedShortTracks4",
                                                                             "RefittedShortTracks5",
                                                                             "RefittedShortTracks6",
                                                                             "RefittedShortTracks7",
                                                                             "RefittedShortTracks8"])
                                                     
shortTrackResolution3to8 = cms.Sequence(SingleLongTrackProducer *
                                        ShortTrackCandidates3 *
                                        ShortTrackCandidates4 *
                                        ShortTrackCandidates5 *
                                        ShortTrackCandidates6 *
                                        ShortTrackCandidates7 *
                                        ShortTrackCandidates8 *
                                        RefittedShortTracks3 *
                                        RefittedShortTracks4 *
                                        RefittedShortTracks5 *
                                        RefittedShortTracks6 *
                                        RefittedShortTracks7 *
                                        RefittedShortTracks8 *
                                        trackingResolution)

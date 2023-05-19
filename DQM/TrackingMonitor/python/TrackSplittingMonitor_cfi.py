# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(track)
#

import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.trackSplittingMonitor_cfi import trackSplittingMonitor
TrackSplitMonitor = trackSplittingMonitor.clone(FolderName = cms.string('TrackSplitMonitoring'),
                                                splitTrackCollection = "splittedTracksP5",
                                                splitMuonCollection = "splitMuons",
                                                ifPlotMuons = True,
                                                pixelHitsPerLeg = 1,
                                                totalHitsPerLeg = 6 ,
                                                d0Cut = 12.0 ,
                                                dzCut = 25.0 ,
                                                ptCut = 4.0 ,
                                                norchiCut = 100.0 ,
                                                ddxyBin = 100 ,
                                                ddxyMin = -200.0 ,
                                                ddxyMax = 200.0 ,
                                                ddzBin = 100,
                                                ddzMin = -400.0,
                                                ddzMax = 400.0,
                                                dphiBin = 100,
                                                dphiMin = -0.01,
                                                dphiMax = 0.01,
                                                dthetaBin = 100,
                                                dthetaMin = -0.01,
                                                dthetaMax = 0.01,
                                                dptBin = 100,
                                                dptMin = -5.0,
                                                dptMax = 5.0,
                                                dcurvBin = 100,
                                                dcurvMin = -0.005,
                                                dcurvMax = 0.005,
                                                normBin = 100,
                                                normMin = -5.0,
                                                normMax = 5.0)

# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(track)
#

import FWCore.ParameterSet.Config as cms

TrackSplitMonitor = cms.EDAnalyzer("TrackSplittingMonitor",
                 
    FolderName = cms.string('TrackSplitMonitoring'),
	
	splitTrackCollection = cms.InputTag("splittedTracksP5"),
	splitMuonCollection = cms.InputTag("splitMuons"),
	ifPlotMuons = cms.bool(True),
	
	pixelHitsPerLeg = cms.int32( 1 ),
	totalHitsPerLeg = cms.int32( 6 ),	
	d0Cut = cms.double( 12.0 ),
	dzCut = cms.double( 25.0 ),	
	ptCut = cms.double( 4.0 ),
	norchiCut = cms.double( 100.0 ),
	
    ddxyBin = cms.int32(100),
    ddxyMin = cms.double(-200.0),
    ddxyMax = cms.double(200.0),

    ddzBin = cms.int32(100),
    ddzMin = cms.double(-400.0),
    ddzMax = cms.double(400.0),

    dphiBin = cms.int32(100),
    dphiMin = cms.double(-0.01),
    dphiMax = cms.double(0.01),

    dthetaBin = cms.int32(100),
    dthetaMin = cms.double(-0.01),
    dthetaMax = cms.double(0.01),

    dptBin = cms.int32(100),
    dptMin = cms.double(-5.0),
    dptMax = cms.double(5.0),

    dcurvBin = cms.int32(100),
    dcurvMin = cms.double(-0.005),
    dcurvMax = cms.double(0.005),

    normBin = cms.int32(100),
    normMin = cms.double(-5.0),
    normMax = cms.double(5.0)
)

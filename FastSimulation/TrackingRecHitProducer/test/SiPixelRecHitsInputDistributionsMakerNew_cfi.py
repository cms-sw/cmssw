import FWCore.ParameterSet.Config as cms

pixRecHitsDistributionsMakerNew = cms.EDAnalyzer("SiPixelRecHitsInputDistributionsMakerNew",
		associatePixel = cms.bool(True),
		associateStrip = cms.bool(False),
		associateRecoTracks = cms.bool(False),
		ROUList = cms.vstring(
			"g4SimHitsTrackerHitsPixelBarrelLowTof",
			"g4SimHitsTrackerHitsPixelBarrelHighTof",
			"g4SimHitsTrackerHitsPixelEndcapLowTof",
			"g4SimHitsTrackerHitsPixelEndcapHighTof"
		),
		outputFile = cms.untracked.string("pixelrechitshistoNew.root"),
		verbose	= cms.untracked.bool(True),
		src	= cms.InputTag("siPixelRecHits")

)

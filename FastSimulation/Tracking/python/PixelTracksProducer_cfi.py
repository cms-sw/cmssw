import FWCore.ParameterSet.Config as cms


pixelTracks = cms.EDProducer("PixelTracksProducer",
                               FilterPSet = cms.PSet(
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        chi2 = cms.double(1000.0),
        nSigmaInvPtTolerance = cms.double(0.0),
        nSigmaTipMaxTolerance = cms.double(0.0),
        ptMin = cms.double(0.1),
        tipMax = cms.double(1.0)
    ),
    FitterPSet = cms.PSet(
        ComponentName = cms.string('PixelFitterByHelixProjections'),
        TTRHBuilder = cms.string('WithoutRefit')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            originHalfLength = cms.double(21.2),
            originRadius = cms.double(0.2),
            originXPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            originZPos = cms.double(0.0),
            precise = cms.bool(True),
            ptMin = cms.double(0.9)
        )
    ),
    SeedProducer = cms.InputTag("pixelTripletSeeds")
)


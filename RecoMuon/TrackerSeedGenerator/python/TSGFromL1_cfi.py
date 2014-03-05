import FWCore.ParameterSet.Config as cms

hltL3TrajectorySeedFromL1 = cms.EDProducer("TSGFromL1Muon",
    FilterPSet = cms.PSet(
        nSigmaInvPtTolerance = cms.double(2.0),
        nSigmaTipMaxTolerance = cms.double(3.0),
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        chi2 = cms.double(1000.0),
        ptMin = cms.double(10.0),
        tipMax = cms.double(0.1)
    ),
    FitterPSet = cms.PSet(
        cotThetaErrorScale = cms.double(1.0),
        tipErrorScale = cms.double(1.0),
        ComponentName = cms.string('L1MuonPixelTrackFitter'),
        invPtErrorScale = cms.double(1.0),
        phiErrorScale = cms.double(1.0),
        zipErrorScale = cms.double(1.0)
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('L1MuonRegionProducer'),
        RegionPSet = cms.PSet(
            originHalfLength = cms.double(15.9),
            originRadius = cms.double(0.1),
            originYPos = cms.double(0.0),
            ptMin = cms.double(10.0),
            originXPos = cms.double(0.0),
            originZPos = cms.double(0.0)
        )
    ),
    L1MuonLabel = cms.InputTag("hltL1extraParticles"),
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('PixelTrackCleanerBySharedHits'),
        diffRelPtCut = cms.double(0.2),
        deltaEtaCut = cms.double(0.01)
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag('PixelLayerPairs')
    )
)




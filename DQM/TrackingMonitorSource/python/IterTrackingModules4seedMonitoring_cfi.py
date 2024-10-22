import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

seedMonitoring = {}

seedMonitoring['initialStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("initialStepSeeds"),
    trackCandInputTag = cms.InputTag("initialStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(100), # could be 50 ?
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(5000),
    clusterLabel      = cms.vstring('Pix'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(20000),
)

seedMonitoring['highPtTripletStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("highPtTripletStepSeeds"),
    trackCandInputTag = cms.InputTag("highPtTripletStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(100),
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(30000),
    clusterLabel      = cms.vstring('Pix'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(20000),
)

seedMonitoring['lowPtQuadStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("lowPtQuadStepSeeds"),
    trackCandInputTag = cms.InputTag("lowPtQuadStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(100),
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(10000),
    clusterLabel      = cms.vstring('Pix'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(20000),
)

seedMonitoring['lowPtTripletStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("lowPtTripletStepSeeds"),
    trackCandInputTag = cms.InputTag("lowPtTripletStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(100),
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(30000),
    clusterLabel      = cms.vstring('Pix'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(20000),
)

seedMonitoring['pixelPairStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("pixelPairStepSeeds"),
    trackCandInputTag = cms.InputTag("pixelPairStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(400),
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(10000),
    TCSizeMax         = cms.double(199.5),
    clusterLabel      = cms.vstring('Pix'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(20000),
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(seedMonitoring['pixelPairStep'],
    RegionSeedingLayersProducer = cms.InputTag("pixelPairStepTrackingRegionsSeedLayersB"),
    RegionSizeBin               = cms.int32(100),
    RegionSizeMax               = cms.double(399.5),
)

seedMonitoring['detachedQuadStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("detachedQuadStepSeeds"),
    trackCandInputTag = cms.InputTag("detachedQuadStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(100),
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(10000),
    TCSizeMax         = cms.double(199.5),
    clusterLabel      = cms.vstring('Pix'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(20000),
)

seedMonitoring['detachedTripletStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("detachedTripletStepSeeds"),
    trackCandInputTag = cms.InputTag("detachedTripletStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(100),
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(30000),
    clusterLabel      = cms.vstring('Pix'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(20000),
)

seedMonitoring['mixedTripletStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("mixedTripletStepSeeds"),
    trackCandInputTag = cms.InputTag("mixedTripletStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(200),
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(10000),
    TCSizeMax         = cms.double(199.5),
    clusterLabel      = cms.vstring('Tot'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(100000),
)

seedMonitoring['pixelLessStep'] = cms.PSet(
    seedInputTag      = cms.InputTag("pixelLessStepSeeds"),
    trackCandInputTag = cms.InputTag("pixelLessStepTrackCandidates"),
    trackSeedSizeBin  = cms.int32(400),
    trackSeedSizeMin  = cms.double(0),
    trackSeedSizeMax  = cms.double(200000),
    clusterLabel      = cms.vstring('Strip'),
    clusterBin        = cms.int32(500),
    clusterMax        = cms.double(100000),
)

seedMonitoring['tobTecStep'] = cms.PSet(
    seedInputTag     = cms.InputTag("tobTecStepSeeds"),
    trackCandInputTag= cms.InputTag("tobTecStepTrackCandidates"),
    trackSeedSizeBin = cms.int32(400),
    trackSeedSizeMin = cms.double(0),
    trackSeedSizeMax = cms.double(100000),
    TCSizeMax        = cms.double(199.5),
    clusterLabel     = cms.vstring('Strip'),
    clusterBin       = cms.int32(100),
    clusterMax       = cms.double(100000),
)

seedMonitoring['displacedGeneralStep'] = cms.PSet(
    seedInputTag     = cms.InputTag("displacedGeneralStepSeeds"),
    trackCandInputTag= cms.InputTag("displacedGeneralStepTrackCandidates"),
    trackSeedSizeBin = cms.int32(400),
    trackSeedSizeMin = cms.double(0),
    trackSeedSizeMax = cms.double(100000),
    TCSizeMax        = cms.double(199.5),
    clusterLabel     = cms.vstring('Strip'),
    clusterBin       = cms.int32(100),
    clusterMax       = cms.double(100000),
)


seedMonitoring['muonSeededStepInOut'] = cms.PSet(
    seedInputTag      = cms.InputTag("muonSeededSeedsInOut"),
    trackCandInputTag = cms.InputTag("muonSeededTrackCandidatesInOut"),
    trackSeedSizeBin  = cms.int32(30),
    trackSeedSizeMin  = cms.double(-0.5),
    trackSeedSizeMax  = cms.double(29.5),
    TCSizeMax         = cms.double(199.5),
    clusterLabel      = cms.vstring('Strip'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(100000),
)

seedMonitoring['muonSeededStepOutIn'] = cms.PSet(
    seedInputTag      = cms.InputTag("muonSeededSeedsOutIn"),
    trackCandInputTag = cms.InputTag("muonSeededTrackCandidatesOutIn"),
    trackSeedSizeBin  = cms.int32(30),
    trackSeedSizeMin  = cms.double(-0.5),
    trackSeedSizeMax  = cms.double(29.5),
    TCSizeMax         = cms.double(199.5),
    clusterLabel      = cms.vstring('Strip'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(100000),
)

seedMonitoring['displacedRegionalStep'] = cms.PSet(
    seedInputTag     = cms.InputTag("displacedRegionalStepSeeds"),
    trackCandInputTag= cms.InputTag("displacedRegionalStepTrackCandidates"),
    trackSeedSizeBin = cms.int32(400),
    trackSeedSizeMin = cms.double(0),
    trackSeedSizeMax = cms.double(100000),
    TCSizeMax        = cms.double(199.5),
    clusterLabel     = cms.vstring('Strip'),
    clusterBin       = cms.int32(100),
    clusterMax       = cms.double(100000),
)

seedMonitoring['muonSeededStepOutInDisplaced'] = cms.PSet(
    seedInputTag      = cms.InputTag("muonSeededSeedsOutInDisplaced"),
    trackCandInputTag = cms.InputTag("muonSeededTrackCandidatesOutInDisplacedg"),
    trackSeedSizeBin  = cms.int32(30),
    trackSeedSizeMin  = cms.double(-0.5),
    trackSeedSizeMax  = cms.double(29.5),
    TCSizeMax         = cms.double(199.5),
    clusterLabel      = cms.vstring('Strip'),
    clusterBin        = cms.int32(100),
    clusterMax        = cms.double(100000),
)

seedMonitoring['jetCoreRegionalStep'] = cms.PSet(
    seedInputTag         = cms.InputTag("jetCoreRegionalStepSeeds"),
    trackCandInputTag    = cms.InputTag("jetCoreRegionalStepTrackCandidates"),
    trackSeedSizeBin     = cms.int32(100),
    trackSeedSizeMin     = cms.double(-0.5),
    trackSeedSizeMax     = cms.double(199.5),
    clusterLabel         = cms.vstring('Tot'),
    clusterBin           = cms.int32(100),
    clusterMax           = cms.double(100000),
    RegionProducer       = cms.InputTag("jetCoreRegionalStepTrackingRegions"),
    RegionCandidates     = cms.InputTag("jetsForCoreTracking"),
    trajCandPerSeedBin   = cms.int32(50),
    trajCandPerSeedMax   = cms.double(49.5),
)

seedMonitoring['jetCoreRegionalStepBarrel'] = cms.PSet(
    seedInputTag         = cms.InputTag("jetCoreRegionalStepSeedsBarrel"),
    trackCandInputTag    = cms.InputTag("jetCoreRegionalStepBarrelTrackCandidates"),
    trackSeedSizeBin     = cms.int32(100),
    trackSeedSizeMin     = cms.double(-0.5),
    trackSeedSizeMax     = cms.double(199.5),
    clusterLabel         = cms.vstring('Tot'),
    clusterBin           = cms.int32(100),
    clusterMax           = cms.double(100000),
    RegionProducer       = cms.InputTag("jetCoreRegionalStepBarrelTrackingRegions"),
    RegionCandidates     = cms.InputTag("jetsForCoreTrackingBarrel"),
    trajCandPerSeedBin   = cms.int32(50),
    trajCandPerSeedMax   = cms.double(49.5),
    doMVAPlots           = cms.bool(True),
    TrackProducerForMVA  = cms.InputTag("jetCoreRegionalStepBarrelTracks"),
    MVAProducers         = cms.InputTag("jetCoreRegionalBarrelStep"),
)

seedMonitoring['jetCoreRegionalStepEndcap'] = cms.PSet(
    seedInputTag         = cms.InputTag("jetCoreRegionalStepSeedsEndcap"),
    trackCandInputTag    = cms.InputTag("jetCoreRegionalStepEndcapTrackCandidates"),
    trackSeedSizeBin     = cms.int32(100),
    trackSeedSizeMin     = cms.double(-0.5),
    trackSeedSizeMax     = cms.double(199.5),
    clusterLabel         = cms.vstring('Tot'),
    clusterBin           = cms.int32(100),
    clusterMax           = cms.double(100000),
    RegionProducer       = cms.InputTag("jetCoreRegionalStepEndcapTrackingRegions"),
    RegionCandidates     = cms.InputTag("jetsForCoreTrackingEndcap"),
    trajCandPerSeedBin   = cms.int32(50),
    trajCandPerSeedMax   = cms.double(49.5),
    doMVAPlots           = cms.bool(True),
    TrackProducerForMVA  = cms.InputTag("jetCoreRegionalStepEndcapTracks"),
    MVAProducers         = cms.InputTag("jetCoreRegionalEndcapStep"),
)

for _eraName, _postfix, _era in _cfg.allEras():
    locals()["selectedIterTrackingStep"+_postfix] = _cfg.iterationAlgos(_postfix)
#selectedIterTrackingStep.append('muonSeededStepOutInDisplaced')


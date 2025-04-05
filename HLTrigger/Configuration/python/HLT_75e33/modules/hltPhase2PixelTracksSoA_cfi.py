import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2@alpaka',
    caGeometry = cms.string('hltPhase2CAGeometry'),
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    ptmin = cms.double(0.9),
    hardCurvCut = cms.double(0.0328407225),
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    fillStatistics = cms.bool(False),
    minHitsPerNtuplet = cms.uint32(4),
    maxNumberOfDoublets = cms.string(str(5*512*1024)),
    maxNumberOfTuples = cms.string(str(32*1024)),
    avgHitsPerTrack = cms.double(5),
    avgCellsPerHit = cms.double(25),
    avgCellsPerCell = cms.double(2),
    avgTracksPerCell = cms.double(1),
    minHitsForSharingCut = cms.uint32(10),
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5.0),
        minPt   = cms.double(0.9),
        maxTip  = cms.double(0.3),
        maxZip  = cms.double(12.),
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

_hltPhase2PixelTracksSoASingleIterPatatrack = hltPhase2PixelTracksSoA.clone( 
            minHitsPerNtuplet = 3,
            avgHitsPerTrack    = 6.5,      
            avgCellsPerHit     = 6, 
            avgCellsPerCell    = 0.151, 
            avgTracksPerCell   = 0.130,
            maxNumberOfDoublets = str(5*512*1024),
            maxNumberOfTuples   = str(256 * 1024)
            )

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
singleIterPatatrack.toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoASingleIterPatatrack)

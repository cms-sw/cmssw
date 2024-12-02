import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    CPE = cms.string('PixelCPEFastParamsPhase2'),
    ptmin = cms.double(0.9),
    CAThetaCutBarrel = cms.double(0.002),
    CAThetaCutForward = cms.double(0.003),
    hardCurvCut = cms.double(0.0328407225),
    dcaCutInnerTriplet = cms.double(0.15),
    dcaCutOuterTriplet = cms.double(0.25),
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    fillStatistics = cms.bool(False),
    minHitsPerNtuplet = cms.uint32(4),
    phiCuts = cms.vint32(
        522, 522, 522, 626, 730, 730, 626, 730, 730, 522, 522,
        522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522,
        522, 522, 522, 522, 522, 522, 522, 730, 730, 730, 730,
        730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730,
        730, 730, 730, 522, 522, 522, 522, 522, 522, 522, 522
    ),
    maxNumberOfDoublets = cms.uint32(5*512*1024),
    minHitsForSharingCut = cms.uint32(10),
    fitNas4 = cms.bool(False),
    doClusterCut = cms.bool(True),
    doZ0Cut = cms.bool(True),
    doPtCut = cms.bool(True),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    idealConditions = cms.bool(False),
    includeJumpingForwardDoublets = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5.0),
        minPt   = cms.double(0.9),
        maxTip  = cms.double(0.3),
        maxZip  = cms.double(12.),
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

_hltPhase2PixelTracksSoASingleIterPatatrack = hltPhase2PixelTracksSoA.clone( minHitsPerNtuplet = 3 )

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
singleIterPatatrack.toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoASingleIterPatatrack)

import FWCore.ParameterSet.Config as cms

hltOTStubProducer = cms.EDProducer(
    "OTStubProducerVectorHitStyle@alpaka",
    otRecHitsSoA=cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    maxWidthBarrelFlat=cms.vdouble(0.0, 0.0575, 0.06, 0.08, 0.09, 0.12, 0.2),
    maxWidthBarrelTilted=cms.vdouble(0.0, 0.0575, 0.06, 0.08, 0.09, 0.12, 0.2),
    maxWidthEndcap=cms.vdouble(0.0, 0.1, 0.1, 0.1, 0.1, 0.1),
    # Per-layer cluster size cuts (999 = disabled)
    #                                     index:  0    1    2    3    4    5    6
    #                                     layer: pad   L1   L2   L3   L4   L5   L6
    maxClusterSizeDiffBarrelFlat=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    maxClusterSizeDiffBarrelTilted=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    maxClusterSizeDiffEndcap=cms.vint32(999, 999, 999, 999, 999, 999),
    maxClusterSizeBarrelFlat=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    maxClusterSizeBarrelTilted=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    maxClusterSizeEndcap=cms.vint32(999, 999, 999, 999, 999, 999),
    # Per-layer cluster size sum cuts (999 = disabled)
    maxClusterSizeBarrelFlatSum=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    maxClusterSizeSumBarrelTilted=cms.vint32(999, 999, 999, 999, 999, 999, 999),
    maxClusterSizeSumEndcap=cms.vint32(999, 999, 999, 999, 999, 999),
    mightGet=cms.optional.untracked.vstring,
    alpaka=cms.untracked.PSet(backend=cms.untracked.string("")),
)

from Configuration.ProcessModifiers.phase2CATrueStubs_cff import phase2CATrueStubs
from Validation.SiTrackerPhase2V.hltTrueStubProducer_cfi import hltTrueStubProducer as _hltOTStubProducerTrue
phase2CATrueStubs.toReplaceWith(hltOTStubProducer, _hltOTStubProducerTrue)

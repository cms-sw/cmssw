import FWCore.ParameterSet.Config as cms

hltPhase2OTVectorHitsSoA = cms.EDProducer(
    "Phase2OTVectorHitsSoAProducer@alpaka",
    otRecHitsSoA=cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    maxWidthBarrelFlat=cms.vdouble(0.0, 0.0575, 0.06, 0.08, 0.09, 0.12, 0.2),
    maxWidthBarrelTilted=cms.vdouble(0.0, 0.0575, 0.06, 0.08, 0.09, 0.12, 0.2),
    maxWidthEndcap=cms.vdouble(0.0, 0.1, 0.1, 0.1, 0.1, 0.1),
    mightGet=cms.optional.untracked.vstring,
    alpaka=cms.untracked.PSet(backend=cms.untracked.string("")),
)

import FWCore.ParameterSet.Config as cms

import RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff
regionalSeedsForL3Isolation = RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff.globalPixelSeeds.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
regionalCandidatesForL3Isolation = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.TrackProducer_cfi
regionalTracksForL3Isolation = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
regionalCKFTracksForL3Isolation = cms.Sequence(regionalSeedsForL3Isolation*regionalCandidatesForL3Isolation*regionalTracksForL3Isolation)
regionalSeedsForL3Isolation.RegionFactoryPSet.ComponentName = 'IsolationRegionAroundL3Muon'
regionalSeedsForL3Isolation.RegionFactoryPSet.RegionPSet = cms.PSet(
    deltaPhiRegion = cms.double(0.24),
    TrkSrc = cms.InputTag("L3Muons"),
    originHalfLength = cms.double(15.0),
    deltaEtaRegion = cms.double(0.24),
    vertexZDefault = cms.double(0.0),
    vertexSrc = cms.string(''),
    originRadius = cms.double(0.2),
    vertexZConstrained = cms.bool(False),
    ptMin = cms.double(1.0)
)
regionalCandidatesForL3Isolation.src = cms.InputTag('regionalSeedsForL3Isolation')
regionalTracksForL3Isolation.src = 'regionalCandidatesForL3Isolation'



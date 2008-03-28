import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cfi import *
regionalSeedsForL3Isolation = copy.deepcopy(globalPixelSeeds)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
regionalCandidatesForL3Isolation = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
regionalTracksForL3Isolation = copy.deepcopy(ctfWithMaterialTracks)
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
regionalCandidatesForL3Isolation.SeedProducer = 'regionalSeedsForL3Isolation'
regionalTracksForL3Isolation.src = 'regionalCandidatesForL3Isolation'


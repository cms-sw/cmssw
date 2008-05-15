import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff import *
import copy
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cfi import *
hltMumukPixelSeedFromL2Candidate = copy.deepcopy(globalPixelSeeds)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
hltCkfTrajectoryFilterMumuk = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
hltCkfTrajectoryBuilderMumuk = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
hltCkfTrackCandidatesMumuk = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
hltCtfWithMaterialTracksMumuk = copy.deepcopy(ctfWithMaterialTracks)
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
hltMumukAllConeTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumuk"),
    particleType = cms.string('mu-')
)

hltMumukTracking = cms.Sequence(hltMumukPixelSeedFromL2Candidate+hltCkfTrackCandidatesMumuk+hltCtfWithMaterialTracksMumuk)
hltMumukCandidates = cms.Sequence(hltMumukAllConeTracks)
l3MumukReco = cms.Sequence(cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("recopixelvertexing")+hltMumukTracking+hltMumukCandidates)
hltMumukPixelSeedFromL2Candidate.RegionFactoryPSet.ComponentName = 'L3MumuTrackingRegion'
hltMumukPixelSeedFromL2Candidate.RegionFactoryPSet.RegionPSet = cms.PSet(
    deltaPhiRegion = cms.double(0.15),
    TrkSrc = cms.InputTag("hltL2Muons"),
    originHalfLength = cms.double(1.0),
    deltaEtaRegion = cms.double(0.15),
    vertexZDefault = cms.double(0.0),
    vertexSrc = cms.string('pixelVertices'),
    originRadius = cms.double(1.0),
    ptMin = cms.double(3.0)
)
hltCkfTrajectoryFilterMumuk.ComponentName = 'hltCkfTrajectoryFilterMumuk'
hltCkfTrajectoryFilterMumuk.filterPset.minPt = 3.0
hltCkfTrajectoryFilterMumuk.filterPset.maxNumberOfHits = 5
hltCkfTrajectoryBuilderMumuk.ComponentName = 'hltCkfTrajectoryBuilderMumuk'
hltCkfTrajectoryBuilderMumuk.trajectoryFilterName = 'hltCkfTrajectoryFilterMumuk'
hltCkfTrajectoryBuilderMumuk.maxCand = 3
hltCkfTrajectoryBuilderMumuk.alwaysUseInvalidHits = False
hltCkfTrackCandidatesMumuk.SeedProducer = 'hltMumukPixelSeedFromL2Candidate'
hltCkfTrackCandidatesMumuk.TrajectoryBuilder = 'hltCkfTrajectoryBuilderMumuk'
hltCtfWithMaterialTracksMumuk.src = 'hltCkfTrackCandidatesMumuk'


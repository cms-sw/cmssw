import FWCore.ParameterSet.Config as cms

from HLTrigger.btau.jetTag.pixelReco_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff import *
import copy
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cfi import *
hltMumuPixelSeedFromL2Candidate = copy.deepcopy(globalPixelSeeds)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
hltCkfTrajectoryFilterMumu = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
hltCkfTrajectoryBuilderMumu = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
hltCkfTrackCandidatesMumu = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
hltCtfWithMaterialTracksMumu = copy.deepcopy(ctfWithMaterialTracks)
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
hltMuTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumu"),
    particleType = cms.string('mu-')
)

Mumutracks = cms.Sequence(hltMumuPixelSeedFromL2Candidate+hltCkfTrackCandidatesMumu+hltCtfWithMaterialTracksMumu)
Mumucand = cms.Sequence(hltMuTracks)
l3displacedMumureco = cms.Sequence(cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+recopixelvertexing+Mumutracks+Mumucand)
hltMumuPixelSeedFromL2Candidate.RegionFactoryPSet.ComponentName = 'L3MumuTrackingRegion'
hltMumuPixelSeedFromL2Candidate.RegionFactoryPSet.RegionPSet = cms.PSet(
    deltaPhiRegion = cms.double(0.15),
    TrkSrc = cms.InputTag("hltL2Muons"),
    originHalfLength = cms.double(1.0),
    deltaEtaRegion = cms.double(0.15),
    vertexZDefault = cms.double(0.0),
    vertexSrc = cms.string('pixelVertices'),
    originRadius = cms.double(1.0),
    ptMin = cms.double(3.0)
)
hltCkfTrajectoryFilterMumu.ComponentName = 'hltCkfTrajectoryFilterMumu'
hltCkfTrajectoryFilterMumu.filterPset.minPt = 3.0
hltCkfTrajectoryFilterMumu.filterPset.maxNumberOfHits = 5
hltCkfTrajectoryBuilderMumu.ComponentName = 'hltCkfTrajectoryBuilderMumu'
hltCkfTrajectoryBuilderMumu.trajectoryFilterName = 'hltCkfTrajectoryFilterMumu'
hltCkfTrajectoryBuilderMumu.maxCand = 3
hltCkfTrajectoryBuilderMumu.alwaysUseInvalidHits = False
hltCkfTrackCandidatesMumu.SeedProducer = 'hltMumuPixelSeedFromL2Candidate'
hltCkfTrackCandidatesMumu.TrajectoryBuilder = 'hltCkfTrajectoryBuilderMumu'
hltCtfWithMaterialTracksMumu.src = 'hltCkfTrackCandidatesMumu'


import FWCore.ParameterSet.Config as cms

from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfBaseTrajectoryFilterP5 = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
CkfTrajectoryBuilderP5 = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
ckfTrackCandidatesP5 = copy.deepcopy(ckfTrackCandidates)
import copy
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
FittingSmootherRKP5 = copy.deepcopy(KFFittingSmoother)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
ctfWithMaterialTracksP5 = copy.deepcopy(ctfWithMaterialTracks)
hltTrackerCosmicsSeedsFilterCTF = cms.EDFilter("HLTCountNumberOfTrajectorySeed",
    src = cms.InputTag("combinatorialcosmicseedfinderP5"),
    MaxN = cms.int32(50),
    MinN = cms.int32(-1)
)

hltTrackerCosmicsTracksFilterCTF = cms.EDFilter("HLTCountNumberOfTrack",
    src = cms.InputTag("ctfWithMaterialTracksP5"),
    MaxN = cms.int32(1000),
    MinN = cms.int32(1)
)

hltTrackerCosmicsSeedsCTF = cms.Sequence(combinatorialcosmicseedfinderP5)
hltTrackerCosmicsTracksCTF = cms.Sequence(ckfTrackCandidatesP5+cms.SequencePlaceholder("offlineBeamSpot")+ctfWithMaterialTracksP5)
combinatorialcosmicseedfinderP5.ClusterCollectionLabel = 'SiStripRawToClustersFacility'
ckfBaseTrajectoryFilterP5.ComponentName = 'ckfBaseTrajectoryFilterP5'
CkfTrajectoryBuilderP5.MeasurementTrackerName = ''
CkfTrajectoryBuilderP5.ComponentName = 'CkfTrajectoryBuilderP5'
CkfTrajectoryBuilderP5.trajectoryFilterName = 'ckfBaseTrajectoryFilterP5'
ckfTrackCandidatesP5.NavigationSchool = 'CosmicNavigationSchool'
ckfTrackCandidatesP5.TrajectoryBuilder = 'CkfTrajectoryBuilderP5'
ckfTrackCandidatesP5.SeedProducer = 'combinatorialcosmicseedfinderP5'
FittingSmootherRKP5.ComponentName = 'FittingSmootherRKP5'
FittingSmootherRKP5.Fitter = 'FitterRK'
FittingSmootherRKP5.Smoother = 'SmootherRK'
FittingSmootherRKP5.MinNumberOfHits = 4
ctfWithMaterialTracksP5.src = 'ckfTrackCandidatesP5'
ctfWithMaterialTracksP5.Fitter = 'FittingSmootherRKP5'
ctfWithMaterialTracksP5.beamSpot = 'offlineBeamSpot'


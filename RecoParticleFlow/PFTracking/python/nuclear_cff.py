import FWCore.ParameterSet.Config as cms

from RecoVertex.NuclearInteractionProducer.NuclearInteraction_cff import *
import copy
from RecoTracker.NuclearSeedGenerator.NuclearSeed_cfi import *
# FIRST NUCLEAR
firstnuclearSeed = copy.deepcopy(nuclearSeed)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
firstnuclearTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
firstnuclearWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoVertex.NuclearInteractionProducer.NuclearInteraction_cfi import *
firstnuclearInteractionMaker = copy.deepcopy(nuclearInteractionMaker)
import copy
from RecoTracker.NuclearSeedGenerator.NuclearSeed_cfi import *
# SECOND NUCLEAR
secondnuclearSeed = copy.deepcopy(nuclearSeed)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
secondnuclearTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
secondnuclearWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoVertex.NuclearInteractionProducer.NuclearInteraction_cfi import *
secondnuclearInteractionMaker = copy.deepcopy(nuclearInteractionMaker)
import copy
from RecoTracker.NuclearSeedGenerator.NuclearSeed_cfi import *
# THIRD NUCLEAR
thirdnuclearSeed = copy.deepcopy(nuclearSeed)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
thirdnuclearTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
thirdnuclearWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoVertex.NuclearInteractionProducer.NuclearInteraction_cfi import *
thirdnuclearInteractionMaker = copy.deepcopy(nuclearInteractionMaker)
import copy
from RecoTracker.NuclearSeedGenerator.NuclearSeed_cfi import *
# second fourth
fourthnuclearSeed = copy.deepcopy(nuclearSeed)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
fourthnuclearTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
fourthnuclearWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoVertex.NuclearInteractionProducer.NuclearInteraction_cfi import *
fourthnuclearInteractionMaker = copy.deepcopy(nuclearInteractionMaker)
firstnuclear = cms.Sequence(firstnuclearSeed*firstnuclearTrackCandidates*firstnuclearWithMaterialTracks*firstnuclearInteractionMaker)
secondnuclear = cms.Sequence(secondnuclearSeed*secondnuclearTrackCandidates*secondnuclearWithMaterialTracks*secondnuclearInteractionMaker)
thirdnuclear = cms.Sequence(thirdnuclearSeed*thirdnuclearTrackCandidates*thirdnuclearWithMaterialTracks*thirdnuclearInteractionMaker)
fourthnuclear = cms.Sequence(fourthnuclearSeed*fourthnuclearTrackCandidates*fourthnuclearWithMaterialTracks*fourthnuclearInteractionMaker)
nuclear = cms.Sequence(firstnuclear*secondnuclear*thirdnuclear*fourthnuclear)
firstnuclearSeed.producer = 'firstvtxFilt'
firstnuclearTrackCandidates.SeedProducer = 'firstnuclearSeed'
firstnuclearTrackCandidates.TrajectoryBuilder = 'nuclearCkfTrajectoryBuilder'
firstnuclearTrackCandidates.RedundantSeedCleaner = 'none'
firstnuclearWithMaterialTracks.src = 'firstnuclearTrackCandidates'
firstnuclearInteractionMaker.primaryProducer = 'firstvtxFilt'
firstnuclearInteractionMaker.seedsProducer = 'firstnuclearSeed'
firstnuclearInteractionMaker.secondaryProducer = 'firstnuclearWithMaterialTracks'
secondnuclearSeed.producer = 'secondvtxFilt'
secondnuclearTrackCandidates.SeedProducer = 'secondnuclearSeed'
secondnuclearTrackCandidates.TrajectoryBuilder = 'nuclearCkfTrajectoryBuilder'
secondnuclearTrackCandidates.RedundantSeedCleaner = 'none'
secondnuclearWithMaterialTracks.src = 'secondnuclearTrackCandidates'
secondnuclearInteractionMaker.primaryProducer = 'secondvtxFilt'
secondnuclearInteractionMaker.seedsProducer = 'secondnuclearSeed'
secondnuclearInteractionMaker.secondaryProducer = 'secondnuclearWithMaterialTracks'
thirdnuclearSeed.producer = 'thirdvtxFilt'
thirdnuclearTrackCandidates.SeedProducer = 'thirdnuclearSeed'
thirdnuclearTrackCandidates.TrajectoryBuilder = 'nuclearCkfTrajectoryBuilder'
thirdnuclearTrackCandidates.RedundantSeedCleaner = 'none'
thirdnuclearWithMaterialTracks.src = 'thirdnuclearTrackCandidates'
thirdnuclearInteractionMaker.primaryProducer = 'thirdvtxFilt'
thirdnuclearInteractionMaker.seedsProducer = 'thirdnuclearSeed'
thirdnuclearInteractionMaker.secondaryProducer = 'thirdnuclearWithMaterialTracks'
fourthnuclearSeed.producer = 'fourthvtxFilt'
fourthnuclearTrackCandidates.SeedProducer = 'fourthnuclearSeed'
fourthnuclearTrackCandidates.TrajectoryBuilder = 'nuclearCkfTrajectoryBuilder'
fourthnuclearTrackCandidates.RedundantSeedCleaner = 'none'
fourthnuclearWithMaterialTracks.src = 'fourthnuclearTrackCandidates'
fourthnuclearInteractionMaker.primaryProducer = 'fourthvtxFilt'
fourthnuclearInteractionMaker.seedsProducer = 'fourthnuclearSeed'
fourthnuclearInteractionMaker.secondaryProducer = 'fourthnuclearWithMaterialTracks'


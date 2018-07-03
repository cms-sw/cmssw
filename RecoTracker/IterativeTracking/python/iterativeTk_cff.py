import FWCore.ParameterSet.Config as cms
from RecoTracker.TkSeedGenerator.trackerClusterCheck_cfi import *

from RecoTracker.IterativeTracking.InitialStepPreSplitting_cff import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *
from RecoTracker.IterativeTracking.JetCoreRegionalStep_cff import *

# Phase1 specific iterations
from RecoTracker.IterativeTracking.HighPtTripletStep_cff import *
from RecoTracker.IterativeTracking.DetachedQuadStep_cff import *
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import *

from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

iterTrackingEarlyTask = _cfg.createEarlyTask("", "", globals())
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(iterTrackingEarlyTask, _cfg.createEarlyTask(_eraName, _postfix, globals()))
iterTrackingEarly = cms.Sequence(iterTrackingEarlyTask)

iterTrackingTask = cms.Task(InitialStepPreSplittingTask,
                            trackerClusterCheck,
                            iterTrackingEarlyTask,
                            earlyGeneralTracks,
                            muonSeededStepTask,
                            preDuplicateMergingGeneralTracks,
                            generalTracksTask,
                            ConvStepTask,
                            conversionStepTracks
                            )
iterTracking = cms.Sequence(iterTrackingTask)

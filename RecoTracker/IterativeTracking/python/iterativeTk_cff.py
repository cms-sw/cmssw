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
from RecoTracker.IterativeTracking.DisplacedGeneralStep_cff import *
from RecoTracker.IterativeTracking.DisplacedRegionalStep_cff import *
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
from RecoTracker.FinalTrackSelectors.trackTfClassifier_cfi import *

trackdnn_source = cms.ESSource("EmptyESSource", recordName = cms.string("TfGraphRecord"), firstValid = cms.vuint32(1), iovIsRunNotTime = cms.bool(True) )
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

from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.ProcessModifiers.displacedRegionalTracking_cff import displacedRegionalTracking
(trackingPhase1 & displacedRegionalTracking).toModify(iterTrackingTask, lambda x: x.add(DisplacedRegionalStepTask))

_iterTrackingTask_trackdnn = iterTrackingTask.copy()
_iterTrackingTask_trackdnn.add(trackdnn_source)                       
trackdnn.toReplaceWith(iterTrackingTask, _iterTrackingTask_trackdnn)
iterTracking = cms.Sequence(iterTrackingTask)

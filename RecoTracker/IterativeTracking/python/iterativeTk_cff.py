import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.FirstStep_cff import *
from RecoTracker.IterativeTracking.SecStep_cff import *
from RecoTracker.IterativeTracking.ThStep_cff import *
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *
iterTracking = cms.Sequence(firstStep*secondStep*thirdStep*fourthStep*fifthStep)



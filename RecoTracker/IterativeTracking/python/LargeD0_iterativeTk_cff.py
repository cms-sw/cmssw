import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.LargeD0_PixelTripletStep_cff import *
from RecoTracker.IterativeTracking.LargeD0_PixelPairStep_cff import *
from RecoTracker.IterativeTracking.LargeD0_PixelTibTidTecStep_cff import *
from RecoTracker.IterativeTracking.LargeD0_TibTidTecStep_cff import *
from RecoTracker.IterativeTracking.LargeD0_TobTecStep_cff import *

largeD0_iterTracking = cms.Sequence(largeD0step1 *
                                    largeD0step2 *
                                    largeD0step3 *
                                    largeD0step4 *
                                    largeD0step5)

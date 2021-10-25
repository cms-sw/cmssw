import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.trackingMkFitCommon_cff import *
from Configuration.ProcessModifiers.trackingMkFitInitialStepPreSplitting_cff import *
from Configuration.ProcessModifiers.trackingMkFitInitialStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitLowPtQuadStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitHighPtTripletStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitLowPtTripletStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitDetachedQuadStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitDetachedTripletStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitPixelPairStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitMixedTripletStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitPixelLessStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitTobTecStep_cff import *

# Use mkFit in selected iterations
trackingMkFit = cms.ModifierChain(
    trackingMkFitCommon,
    trackingMkFitInitialStepPreSplitting,
    trackingMkFitInitialStep,
#    trackingMkFitLowPtQuadStep,       # to be enabled later
    trackingMkFitHighPtTripletStep,
#    trackingMkFitLowPtTripletStep,    # to be enabled later
    trackingMkFitDetachedQuadStep,
#    trackingMkFitDetachedTripletStep, # to be enabled later
#    trackingMkFitPixelPairStep,       # to be enabled later
#    trackingMkFitMixedTripletStep,    # to be enabled later
#    trackingMkFitPixelLessStep,       # to be enabled later
#    trackingMkFitTobTecStep,          # to be enabled later
)

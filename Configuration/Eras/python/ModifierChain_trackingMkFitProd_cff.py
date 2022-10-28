import FWCore.ParameterSet.Config as cms

# to replace CKF with MkFit in select iterations
from Configuration.ProcessModifiers.trackingMkFitCommon_cff import *
from Configuration.ProcessModifiers.trackingMkFitInitialStepPreSplitting_cff import *
from Configuration.ProcessModifiers.trackingMkFitInitialStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitHighPtTripletStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitDetachedQuadStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitDetachedTripletStep_cff import *

trackingMkFitProd =  cms.ModifierChain(
    trackingMkFitCommon,
    trackingMkFitInitialStepPreSplitting,
    trackingMkFitInitialStep,
    trackingMkFitHighPtTripletStep,
    trackingMkFitDetachedQuadStep,
    trackingMkFitDetachedTripletStep,
)

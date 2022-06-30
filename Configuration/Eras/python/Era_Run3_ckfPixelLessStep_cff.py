import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.trackingMkFitPixelLessStep_cff import *
from Configuration.Eras.Era_Run3_cff import Run3

Run3_ckfPixelLessStep = cms.ModifierChain(Run3.copyAndExclude([trackingMkFitPixelLessStep]))

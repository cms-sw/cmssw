import FWCore.ParameterSet.Config as cms
from FastSimulation.Validation.globalValidation_cff import *
from HLTriggerOffline.Common.HLTValidation_cff import *

validation = cms.Sequence(globalValidation+hltvalidation_fastsim) 

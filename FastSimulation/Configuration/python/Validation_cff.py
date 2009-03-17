import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Common.HLTValidation_cff import *

validation = cms.Sequence(hltvalidation_fastsim) 

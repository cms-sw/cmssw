import FWCore.ParameterSet.Config as cms

from ..tasks.HLTBeamSpotTask_cfi import *

HLTBeamSpot = cms.Sequence(HLTBeamSpotTask)

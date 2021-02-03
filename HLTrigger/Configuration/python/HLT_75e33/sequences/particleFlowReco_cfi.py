import FWCore.ParameterSet.Config as cms

from ..tasks.particleFlowRecoTask_cfi import *

particleFlowReco = cms.Sequence(particleFlowRecoTask)

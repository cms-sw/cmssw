import FWCore.ParameterSet.Config as cms

from ..tasks.HLTParticleFlowTask_cfi import *

HLTParticleFlowSequence = cms.Sequence(HLTParticleFlowTask)

import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfPileUp_cfi  import *
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import *

pfNoPileUpTask = cms.Task(
    pfPileUp,
    pfNoPileUp 
    )
pfNoPileUpSequence = cms.Sequence(pfNoPileUpTask)

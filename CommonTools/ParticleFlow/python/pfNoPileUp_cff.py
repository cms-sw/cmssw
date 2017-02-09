import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfPileUp_cfi  import *
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import *

pfNoPileUpSequence = cms.Sequence(
    pfPileUp +
    pfNoPileUp 
    )

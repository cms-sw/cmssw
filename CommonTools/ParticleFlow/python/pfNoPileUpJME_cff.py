import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfPileUp_cfi  import *
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import *

pfPileUpJME = pfPileUp.clone( checkClosestZVertex = cms.bool(False) )
pfNoPileUpJME = pfNoPileUp.clone( topCollection = 'pfPileUpJME')

pfNoPileUpJMESequence = cms.Sequence(
    pfPileUpJME +
    pfNoPileUpJME
    )

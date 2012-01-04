import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfPileUp_cfi  import *
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import *

pfPileUpIso = pfPileUp.clone(PFCandidates = 'particleFlow' )
pfNoPileUpIso = pfNoPileUp.clone( topCollection = 'pfPileUpIso',
                                  bottomCollection='particleFlow')

pfNoPileUpIsoSequence = cms.Sequence(
    pfPileUpIso +
    pfNoPileUpIso
    )

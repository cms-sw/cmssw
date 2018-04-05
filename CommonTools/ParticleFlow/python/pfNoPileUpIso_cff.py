import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfPileUp_cfi  import pfPileUp as _pfPileUp
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import pfNoPileUp as _pfNoPileUp

pfPileUpIso = _pfPileUp.clone()
pfNoPileUpIso = _pfNoPileUp.clone( topCollection = 'pfPileUpIso')

pfNoPileUpIsoTask = cms.Task(
    pfPileUpIso,
    pfNoPileUpIso
    )
pfNoPileUpIsoSequence = cms.Sequence(pfNoPileUpIsoTask)

from CommonTools.ParticleFlow.PFBRECO_cff import *

pfParticleSelectionForIsoSequence = cms.Sequence(
    particleFlowPtrs +
    pfParticleSelectionPFBRECOSequence
    )
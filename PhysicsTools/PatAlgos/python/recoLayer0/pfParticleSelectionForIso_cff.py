from RecoParticleFlow.PFProducer.pfLinker_cff import particleFlowPtrs
from CommonTools.ParticleFlow.pfParticleSelection_cff import *
pfPileUp.PFCandidates = 'particleFlowPtrs'
pfNoPileUp.bottomCollection = 'particleFlowPtrs'
pfPileUpIso.PFCandidates = 'particleFlowPtrs'
pfNoPileUpIso.bottomCollection='particleFlowPtrs'
pfPileUpJME.PFCandidates = 'particleFlowPtrs'
pfNoPileUpJME.bottomCollection='particleFlowPtrs'

pfParticleSelectionForIsoSequence = cms.Sequence(
    particleFlowPtrs +
    pfNoPileUpIsoSequence +
    pfParticleSelectionSequence
    )
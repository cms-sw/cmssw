import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cfi import *
from CommonTools.ParticleFlow.pfNoPileUpJME_cff import *
from CommonTools.ParticleFlow.PFBRECO_cff import pfPileUpPFBRECO, pfNoPileUpPFBRECO

packedPFCandidatesTask = cms.Task(
    packedPFCandidates,
    pfNoPileUpJMETask,
    pfPileUpPFBRECO,
    pfNoPileUpPFBRECO
)

from RecoHI.HiJetAlgos.HiBadParticleCleaner_cfi import cleanedParticleFlow

packedPFCandidatesRemoved = packedPFCandidates.clone(
    inputCollection = "cleanedParticleFlow:removed",
    vertexAssociator = "primaryVertexAssociationCleaned:original"
)

_pp_on_AA_2018_packedPFCandidatesTask = cms.Task(cleanedParticleFlow,packedPFCandidatesRemoved,packedPFCandidatesTask.copy())
from Configuration.ProcessModifiers.run2_miniAOD_pp_on_AA_103X_cff import run2_miniAOD_pp_on_AA_103X
run2_miniAOD_pp_on_AA_103X.toReplaceWith(packedPFCandidatesTask,_pp_on_AA_2018_packedPFCandidatesTask)

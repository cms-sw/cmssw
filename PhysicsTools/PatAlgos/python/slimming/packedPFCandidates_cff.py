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

from RecoHI.HiJetAlgos.HiBadParticleFilter_cfi import filteredParticleFlow

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
packedPFCandidatesCleaned = packedPFCandidates.clone(
    inputCollection = "filteredParticleFlow:cleaned",
    vertexAssociator = "primaryVertexAssociationCleaned:original"
)

_pp_on_AA_2018_packedPFCandidatesTask = cms.Task(filteredParticleFlow,packedPFCandidatesCleaned,packedPFCandidatesTask.copy())
pp_on_AA_2018.toReplaceWith(packedPFCandidatesTask,_pp_on_AA_2018_packedPFCandidatesTask)

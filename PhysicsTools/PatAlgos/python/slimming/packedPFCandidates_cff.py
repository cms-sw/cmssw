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

import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.PFBRECO_cff import pfPileUpIsoPFBRECO, pfNoPileUpIsoPFBRECO, pfNoPileUpIsoPFBRECOSequence, pfNoPileUpIsoPFBRECOTask
from CommonTools.ParticleFlow.PFBRECO_cff import pfAllNeutralHadronsPFBRECO, pfAllChargedHadronsPFBRECO, pfAllPhotonsPFBRECO, pfAllChargedParticlesPFBRECO, pfPileUpAllChargedParticlesPFBRECO, pfAllNeutralHadronsAndPhotonsPFBRECO, pfSortByTypePFBRECOSequence, pfSortByTypePFBRECOTask

patPFCandidateIsoDepositSelectionTask = cms.Task(
       pfNoPileUpIsoPFBRECOTask,
       pfSortByTypePFBRECOTask
       )
patPFCandidateIsoDepositSelection = cms.Sequence(patPFCandidateIsoDepositSelectionTask)

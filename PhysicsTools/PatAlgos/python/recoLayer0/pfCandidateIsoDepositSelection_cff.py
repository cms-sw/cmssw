import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.PFBRECO_cff import pfPileUpIsoPFBRECO, pfNoPileUpIsoPFBRECO, pfNoPileUpIsoPFBRECOSequence
from CommonTools.ParticleFlow.PFBRECO_cff import pfAllNeutralHadronsPFBRECO, pfAllChargedHadronsPFBRECO, pfAllPhotonsPFBRECO, pfAllChargedParticlesPFBRECO, pfPileUpAllChargedParticlesPFBRECO, pfAllNeutralHadronsAndPhotonsPFBRECO, pfSortByTypePFBRECOSequence

patPFCandidateIsoDepositSelection = cms.Sequence(
       pfNoPileUpIsoPFBRECOSequence +
       pfSortByTypePFBRECOSequence
       )

import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import *
from CommonTools.ParticleFlow.pfNoPileUp_cff import *

pfPileUp.PFCandidates       = 'particleFlow'
pfNoPileUp.bottomCollection = 'particleFlow'

patPFCandidateIsoDepositSelection = cms.Sequence(
    pfNoPileUpSequence *
    ( pfAllNeutralHadrons +
      pfAllChargedHadrons +
      pfAllPhotons )
    )





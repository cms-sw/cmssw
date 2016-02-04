import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import *
from CommonTools.ParticleFlow.pfNoPileUp_cff import *


patPFCandidateIsoDepositSelection = cms.Sequence(
    pfNoPileUpSequence *
    ( pfAllNeutralHadrons +
      pfAllChargedHadrons +
      pfAllPhotons )
    )





import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import *
from CommonTools.ParticleFlow.pfNoPileUpIso_cff  import *

pfPileUpIso.PFCandidates = 'particleFlow'
pfNoPileUpIso.bottomCollection='particleFlow'

patPFCandidateIsoDepositSelection = cms.Sequence(
       pfNoPileUpIsoSequence +
       pfSortByTypeSequence
       )

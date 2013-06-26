import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import *
from CommonTools.ParticleFlow.pfNoPileUpIso_cff  import *

pfPileUpIso.PFCandidates = 'particleFlowPtrs'
pfNoPileUpIso.bottomCollection='particleFlowPtrs'

patPFCandidateIsoDepositSelection = cms.Sequence(
       pfNoPileUpIsoSequence +
       pfSortByTypeSequence
       )

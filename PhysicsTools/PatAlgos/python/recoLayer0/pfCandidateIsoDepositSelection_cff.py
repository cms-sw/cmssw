import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.PFBRECO_cff  import *

patPFCandidateIsoDepositSelection = cms.Sequence(
       pfNoPileUpIsoPFBRECOSequence +
       pfSortByTypePFBRECOSequence
       )

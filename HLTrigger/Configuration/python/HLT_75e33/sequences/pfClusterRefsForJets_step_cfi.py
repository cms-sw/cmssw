import FWCore.ParameterSet.Config as cms

from ..tasks.pfClusterRefsForJets_stepTask_cfi import *

pfClusterRefsForJets_step = cms.Sequence(pfClusterRefsForJets_stepTask)

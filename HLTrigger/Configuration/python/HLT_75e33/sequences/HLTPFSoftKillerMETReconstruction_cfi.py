import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowSoftKiller_cfi import *
from ..modules.hltPFSoftKillerMET_cfi import *

HLTPFSoftKillerMETReconstruction = cms.Sequence(hltParticleFlowSoftKiller+hltPFSoftKillerMET)

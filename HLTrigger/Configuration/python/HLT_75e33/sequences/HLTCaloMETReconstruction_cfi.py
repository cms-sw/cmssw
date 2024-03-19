import FWCore.ParameterSet.Config as cms

from ..modules.hltCaloMET_cfi import *

HLTCaloMETReconstruction = cms.Sequence(hltCaloMET)

import FWCore.ParameterSet.Config as cms

from ..sequences.particleFlowReco_cfi import *

highlevelreco = cms.Sequence(particleFlowReco)

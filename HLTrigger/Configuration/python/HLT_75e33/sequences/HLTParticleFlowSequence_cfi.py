import FWCore.ParameterSet.Config as cms

from ..sequences.globalreco_cfi import *
from ..sequences.highlevelreco_cfi import *
from ..sequences.localreco_cfi import *
from ..sequences.RawToDigi_cfi import *

HLTParticleFlowSequence = cms.Sequence(RawToDigi+localreco+globalreco+highlevelreco)

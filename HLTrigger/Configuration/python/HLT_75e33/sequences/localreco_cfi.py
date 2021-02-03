import FWCore.ParameterSet.Config as cms

from ..modules.bunchSpacingProducer_cfi import *
from ..sequences.calolocalreco_cfi import *
from ..sequences.muonlocalreco_cfi import *
from ..sequences.trackerlocalreco_cfi import *

localreco = cms.Sequence(bunchSpacingProducer+calolocalreco+muonlocalreco+trackerlocalreco+cms.Sequence(cms.Task()))

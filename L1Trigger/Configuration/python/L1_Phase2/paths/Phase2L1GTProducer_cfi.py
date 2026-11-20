import FWCore.ParameterSet.Config as cms

from ..sequences.l1tGTProducerSequence_cfi import *

Phase2L1GTProducer = cms.Path(l1tGTProducerSequence)

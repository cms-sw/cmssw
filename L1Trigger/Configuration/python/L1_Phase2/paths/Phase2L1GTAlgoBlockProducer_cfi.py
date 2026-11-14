import FWCore.ParameterSet.Config as cms

from ..sequences.l1tGTAlgoBlockProducerSequence_cfi import *

Phase2L1GTAlgoBlockProducer = cms.Path(l1tGTAlgoBlockProducerSequence)

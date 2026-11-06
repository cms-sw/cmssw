import FWCore.ParameterSet.Config as cms

from ..modules.l1tGTProducer_cfi import *

l1tGTProducerSequence = cms.Sequence(l1tGTProducer)

import FWCore.ParameterSet.Config as cms

from ..sequences.L1TPFJetsExtendedTask_cfi import *
from ..modules.l1tTOoLLiPProducer_cfi import *
from ..modules.l1tTOoLLiPProducerCorrectedEmulator_cfi import *

L1TTOoLLiPTask = cms.Task(L1TPFJetsExtendedTask, l1tTOoLLiPProducer, l1tTOoLLiPProducerCorrectedEmulator)

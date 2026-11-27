import FWCore.ParameterSet.Config as cms

from ..sequences.L1TPFJetsExtendedTask_cfi import *
from ..modules.l1tBJetProducerPuppi_cfi import *
from ..modules.l1tBJetProducerPuppiCorrectedEmulator_cfi import *

L1TBJetsTask = cms.Task(L1TPFJetsExtendedTask, l1tBJetProducerPuppi, l1tBJetProducerPuppiCorrectedEmulator)

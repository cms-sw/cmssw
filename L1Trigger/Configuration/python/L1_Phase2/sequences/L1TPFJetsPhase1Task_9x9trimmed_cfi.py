import FWCore.ParameterSet.Config as cms

from ..modules.l1tPhase1JetCalibrator9x9trimmed_cfi import *
from ..modules.l1tPhase1JetProducer9x9trimmed_cfi import *
from ..modules.l1tPhase1JetSumsProducer9x9trimmed_cfi import *

L1TPFJetsPhase1Task_9x9trimmed = cms.Task(l1tPhase1JetCalibrator9x9trimmed, l1tPhase1JetProducer9x9trimmed, l1tPhase1JetSumsProducer9x9trimmed)

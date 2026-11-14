import FWCore.ParameterSet.Config as cms

from ..modules.l1tPhase1JetCalibrator9x9_cfi import *
from ..modules.l1tPhase1JetProducer9x9_cfi import *
from ..modules.l1tPhase1JetSumsProducer9x9_cfi import *

L1TPFJetsPhase1Task_9x9 = cms.Task(l1tPhase1JetCalibrator9x9, l1tPhase1JetProducer9x9, l1tPhase1JetSumsProducer9x9)

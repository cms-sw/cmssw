import FWCore.ParameterSet.Config as cms

from ..tasks.HLTGsfElectronL1SeededTask_cfi import *

HLTGsfElectronL1SeededSequence = cms.Sequence(HLTGsfElectronL1SeededTask)

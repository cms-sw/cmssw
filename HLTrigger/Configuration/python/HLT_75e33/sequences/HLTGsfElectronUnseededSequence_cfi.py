import FWCore.ParameterSet.Config as cms

from ..tasks.HLTGsfElectronUnseededTask_cfi import *

HLTGsfElectronUnseededSequence = cms.Sequence(HLTGsfElectronUnseededTask)

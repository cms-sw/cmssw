import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TECALTPG_unpack_cff import *
from DQM.L1TMonitor.L1TECALTPG_cfi import *
l1tecaltpgpath = cms.Path(ecalBarrelDataSequence*l1tecaltpg)


import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TGEM.simMuonGEMPadDigis_cfi import *
from L1Trigger.L1TGEM.simMuonGEMPadDigiClusters_cfi import *

simMuonGEMPadTask = cms.Task(simMuonGEMPadDigis, simMuonGEMPadDigiClusters)

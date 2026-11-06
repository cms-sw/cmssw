import FWCore.ParameterSet.Config as cms

from ..modules.simMuonGEMPadDigiClusters_cfi import *
from ..modules.simMuonGEMPadDigis_cfi import *

simMuonGEMPadTask = cms.Task(simMuonGEMPadDigiClusters, simMuonGEMPadDigis)

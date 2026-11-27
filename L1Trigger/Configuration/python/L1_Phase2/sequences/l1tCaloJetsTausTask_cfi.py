import FWCore.ParameterSet.Config as cms

from ..sequences.l1tHGCalEnergySplitTowersTask_cfi import *
from ..modules.l1tPhase2CaloJetEmulator_cfi import *

l1tCaloJetsTausTask = cms.Task(l1tHGCalEnergySplitTowersTask, l1tPhase2CaloJetEmulator)

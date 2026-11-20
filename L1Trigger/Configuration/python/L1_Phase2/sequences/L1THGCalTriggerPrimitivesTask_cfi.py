import FWCore.ParameterSet.Config as cms

from ..sequences.L1THGCalBackEndLayer1_cfi import *
from ..sequences.L1THGCalBackEndLayer2_cfi import *
from ..sequences.L1THGCalConcentrator_cfi import *
from ..sequences.L1THGCalTower_cfi import *
from ..sequences.L1THGCalTowerMap_cfi import *
from ..sequences.L1THGCalVFE_cfi import *

L1THGCalTriggerPrimitivesTask = cms.Task(L1THGCalBackEndLayer1, L1THGCalBackEndLayer2, L1THGCalConcentrator, L1THGCalTower, L1THGCalTowerMap, L1THGCalVFE)

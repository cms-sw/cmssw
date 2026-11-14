import FWCore.ParameterSet.Config as cms

from ..modules.l1tHGCalEnergySplitTowerMapProducer_cfi import *
from ..modules.l1tHGCalEnergySplitTowerProducer_cfi import *

l1tHGCalEnergySplitTowersTask = cms.Task(l1tHGCalEnergySplitTowerMapProducer, l1tHGCalEnergySplitTowerProducer)

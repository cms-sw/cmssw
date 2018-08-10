import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.simDigis_cff import *

# define a core which can be extended in customizations:
SimL1CaloEmulator = cms.Sequence( SimL1TCalorimeter )

# Emulators are configured from DB (GlobalTags)
# but in the integration branch conffigure from static hackConditions
from L1Trigger.L1TCalorimeter.hackConditions_cff import *

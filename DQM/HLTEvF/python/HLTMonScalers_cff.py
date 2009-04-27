import FWCore.ParameterSet.Config as cms

from EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi import *
from DQM.TrigXMonitor.L1TScalersSCAL_cfi import *
hltScalRawToDigi = cms.Path(scalersRawToDigi)
hltMonScal = cms.EndPath(l1tscalers)


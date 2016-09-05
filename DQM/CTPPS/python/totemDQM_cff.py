import FWCore.ParameterSet.Config as cms

from DQM.CTPPS.totemDAQTriggerDQMSource_cfi import *

from DQM.CTPPS.totemRPDQMSource_cfi import *

from DQM.CTPPS.totemRPDQMHarvester_cfi import *

totemDQM = cms.Sequence(
  totemDAQTriggerDQMSource *
  totemRPDQMSource *
  totemRPDQMHarvester
)

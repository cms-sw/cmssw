import FWCore.ParameterSet.Config as cms

from DQM.CTPPS.totemDAQTriggerDQMSource_cfi import *

from DQM.CTPPS.totemRPDQMHarvester_cfi import *
from DQM.CTPPS.totemRPDQMSource_cfi import *

from DQM.CTPPS.ctppsPixelDQMSource_cfi import *

ctppsDQM = cms.Sequence(
  totemDAQTriggerDQMSource
 *(totemRPDQMSource + ctppsPixelDQMSource)
 *totemRPDQMHarvester
)

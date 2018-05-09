import FWCore.ParameterSet.Config as cms

from DQM.CTPPS.totemDAQTriggerDQMSource_cfi import *

from DQM.CTPPS.totemRPDQMHarvester_cfi import *
from DQM.CTPPS.totemRPDQMSource_cfi import *

from DQM.CTPPS.ctppsDiamondDQMSource_cfi import *

from DQM.CTPPS.totemTimingDQMSource_cfi import *

from DQM.CTPPS.ctppsPixelDQMSource_cfi import *

from DQM.CTPPS.elasticPlotDQMSource_cfi import *

from DQM.CTPPS.ctppsCommonDQMSource_cfi import *

ctppsDQM = cms.Sequence(
    totemDAQTriggerDQMSource
    + (totemRPDQMSource * totemRPDQMHarvester)
    + ctppsDiamondDQMSource
    + totemTimingDQMSource
    + ctppsPixelDQMSource
    + ctppsCommonDQMSource
)

ctppsDQMElastic = cms.Sequence(
    totemDAQTriggerDQMSource
    + (totemRPDQMSource * totemRPDQMHarvester)
    + ctppsDiamondDQMSource
    + totemTimingDQMSource
    + ctppsPixelDQMSource
    + ctppsCommonDQMSource
    + elasticPlotDQMSource
)

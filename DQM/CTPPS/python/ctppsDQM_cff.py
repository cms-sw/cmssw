import FWCore.ParameterSet.Config as cms

from DQM.CTPPS.totemDAQTriggerDQMSource_cfi import *

from DQM.CTPPS.totemRPDQMHarvester_cfi import *
from DQM.CTPPS.totemRPDQMSource_cfi import *

from DQM.CTPPS.ctppsDiamondDQMSource_cfi import *

from DQM.CTPPS.ctppsPixelDQMSource_cfi import *

from DQM.CTPPS.elasticPlotDQMSource_cfi import *

ctppsDQM = cms.Sequence(
    totemDAQTriggerDQMSource
    + (totemRPDQMSource * totemRPDQMHarvester)
    + ctppsDiamondDQMSource
    + ctppsPixelDQMSource
)

ctppsDQMElastic = cms.Sequence(
    totemDAQTriggerDQMSource
    + (totemRPDQMSource * totemRPDQMHarvester)
    + ctppsDiamondDQMSource
    + ctppsPixelDQMSource
    + elasticPlotDQMSource
)

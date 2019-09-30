import FWCore.ParameterSet.Config as cms

from DQM.CTPPS.totemDAQTriggerDQMSource_cfi import *

from DQM.CTPPS.totemRPDQMHarvester_cfi import *
from DQM.CTPPS.totemRPDQMSource_cfi import *

from DQM.CTPPS.ctppsDiamondDQMSource_cfi import *

from DQM.CTPPS.totemTimingDQMSource_cfi import *

from DQM.CTPPS.ctppsPixelDQMSource_cfi import *

from DQM.CTPPS.elasticPlotDQMSource_cfi import *

from DQM.CTPPS.ctppsCommonDQMSource_cfi import *

ctppsDQM = cms.Sequence()
ctppsDQMElastic = cms.Sequence()
ctppsDQMHarvest = cms.Sequence()

_ctppsDQM = ctppsDQM.copy()
_ctppsDQMElastic = ctppsDQMElastic.copy()
_ctppsDQMHarvest = ctppsDQMHarvest.copy()

_ctppsDQM = cms.Sequence(
    totemDAQTriggerDQMSource
    + totemRPDQMSource
    + ctppsDiamondDQMSource
    + totemTimingDQMSource
    + ctppsPixelDQMSource
    + ctppsCommonDQMSource
)

_ctppsDQMElastic = cms.Sequence(
    totemDAQTriggerDQMSource
    + totemRPDQMSource
    + ctppsDiamondDQMSource
    + totemTimingDQMSource
    + ctppsPixelDQMSource
    + ctppsCommonDQMSource
    + elasticPlotDQMSource
)

_ctppsDQMHarvest = cms.Sequence(totemRPDQMHarvester)

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toReplaceWith(ctppsDQM, _ctppsDQM)
ctpps_2016.toReplaceWith(ctppsDQMElastic, _ctppsDQMElastic)
ctpps_2016.toReplaceWith(ctppsDQMHarvest, _ctppsDQMHarvest)

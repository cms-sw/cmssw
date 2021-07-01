import FWCore.ParameterSet.Config as cms

from DQM.CTPPS.totemDAQTriggerDQMSource_cfi import *

from DQM.CTPPS.totemRPDQMHarvester_cfi import *
from DQM.CTPPS.totemRPDQMSource_cfi import *

from DQM.CTPPS.ctppsDiamondDQMSource_cfi import *

from DQM.CTPPS.totemTimingDQMSource_cfi import *

from DQM.CTPPS.ctppsPixelDQMSource_cfi import *

from DQM.CTPPS.elasticPlotDQMSource_cfi import *

from DQM.CTPPS.ctppsCommonDQMSource_cfi import *

# sequences used by the online DQM in normal running
ctppsCommonDQMSourceOnline = ctppsCommonDQMSource.clone(
  makeProtonRecoPlots = False
)

_ctppsDQMOnlineSource = cms.Sequence(
  ctppsPixelDQMSource
  + ctppsDiamondDQMSource
  + totemTimingDQMSource
  + ctppsCommonDQMSourceOnline
)

_ctppsDQMOnlineHarvest = cms.Sequence(
)

# sequences used by the online DQM in calibration mode
_ctppsDQMCalibrationSource = cms.Sequence(
  totemRPDQMSource
  + ctppsPixelDQMSource
  + ctppsDiamondDQMSource
  + totemTimingDQMSource
  + ctppsCommonDQMSourceOnline
  + elasticPlotDQMSource
)

_ctppsDQMCalibrationHarvest = cms.Sequence(
  totemRPDQMHarvester
)

# sequences used by the offline DQM
ctppsCommonDQMSourceOffline = ctppsCommonDQMSource.clone(
  makeProtonRecoPlots = True
)

#Check if perLSsaving is enabled to mask MEs vs LS
from DQMServices.Core.DQMStore_cfi import DQMStore
if(DQMStore.saveByLumi):
    ctppsDiamondDQMSource.perLSsaving=True
    totemTimingDQMSource.perLSsaving=True
    ctppsCommonDQMSourceOffline.perLSsaving=True

_ctppsDQMOfflineSource = cms.Sequence(
  ctppsPixelDQMOfflineSource
  + ctppsDiamondDQMSource
  + totemTimingDQMSource
  + ctppsCommonDQMSourceOffline
)

_ctppsDQMOfflineHarvest = cms.Sequence(
)

# the actually used sequences must be empty for pre-PPS data
from Configuration.Eras.Modifier_ctpps_cff import ctpps

ctppsDQMOnlineSource = cms.Sequence()
ctppsDQMOnlineHarvest = cms.Sequence()
ctpps.toReplaceWith(ctppsDQMOnlineSource, _ctppsDQMOnlineSource)
ctpps.toReplaceWith(ctppsDQMOnlineHarvest, _ctppsDQMOnlineHarvest)

ctppsDQMCalibrationSource = cms.Sequence()
ctppsDQMCalibrationHarvest = cms.Sequence()
ctpps.toReplaceWith(ctppsDQMCalibrationSource, _ctppsDQMCalibrationSource)
ctpps.toReplaceWith(ctppsDQMCalibrationHarvest, _ctppsDQMCalibrationHarvest)

ctppsDQMOfflineSource = cms.Sequence()
ctppsDQMOfflineHarvest = cms.Sequence()
ctpps.toReplaceWith(ctppsDQMOfflineSource, _ctppsDQMOfflineSource)
ctpps.toReplaceWith(ctppsDQMOfflineHarvest, _ctppsDQMOfflineHarvest)

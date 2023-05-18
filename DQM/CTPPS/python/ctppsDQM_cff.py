import FWCore.ParameterSet.Config as cms

from DQM.CTPPS.totemDAQTriggerDQMSource_cfi import *

from DQM.CTPPS.totemRPDQMHarvester_cfi import *
from DQM.CTPPS.totemRPDQMSource_cfi import *

from DQM.CTPPS.ctppsDiamondDQMSource_cfi import *

from DQM.CTPPS.diamondSampicDQMSource_cfi import *

from DQM.CTPPS.totemTimingDQMSource_cfi import *

from DQM.CTPPS.ctppsPixelDQMSource_cfi import *

from DQM.CTPPS.elasticPlotDQMSource_cfi import *

from DQM.CTPPS.ctppsCommonDQMSource_cfi import *

from DQM.CTPPS.ctppsRandomDQMSource_cfi import *

# sequences used by the online DQM in normal running
ctppsCommonDQMSourceOnline = ctppsCommonDQMSource.clone(
  makeProtonRecoPlots = False
)

_ctppsDQMOnlineSource = cms.Sequence(
  ctppsPixelDQMSource
  + ctppsDiamondDQMSource
  + diamondSampicDQMSourceOnline
  + ctppsCommonDQMSourceOnline
)

_ctppsDQMOnlineHarvest = cms.Sequence(
)

# sequences used by the online DQM in calibration mode
_ctppsDQMCalibrationSource = cms.Sequence(
  totemRPDQMSource
  + ctppsPixelDQMSource
  + ctppsDiamondDQMSource
  + diamondSampicDQMSourceOnline
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

# sequences used by the dedicated random trigger stream
_ctppsDQMRandomSource = cms.Sequence(
  ctppsRandomDQMSource
)

_ctppsDQMRandomHarvest = cms.Sequence(
)

#Check if perLSsaving is enabled to mask MEs vs LS
from Configuration.ProcessModifiers.dqmPerLSsaving_cff import dqmPerLSsaving
dqmPerLSsaving.toModify(ctppsDiamondDQMSource, perLSsaving=True)
dqmPerLSsaving.toModify(diamondSampicDQMSourceOffline, perLSsaving=True)
dqmPerLSsaving.toModify(ctppsCommonDQMSourceOffline, perLSsaving=True)
dqmPerLSsaving.toModify(ctppsDiamondDQMOfflineSource, perLSsaving=True)
dqmPerLSsaving.toModify(totemTimingDQMSource, perLSsaving=True)

_ctppsDQMOfflineSource = cms.Sequence(
  ctppsPixelDQMOfflineSource
  + ctppsDiamondDQMOfflineSource
  + diamondSampicDQMSourceOffline
  + ctppsCommonDQMSourceOffline
)

_ctppsDQMOfflineHarvest = cms.Sequence(
)

from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
ctpps_2018.toReplaceWith(
    _ctppsDQMOfflineSource,
    cms.Sequence(
	  ctppsPixelDQMOfflineSource
	  + ctppsDiamondDQMOfflineSource
	  + totemTimingDQMSource
	  + ctppsCommonDQMSourceOffline
    )
    
)

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toReplaceWith(
    _ctppsDQMOfflineSource,
    cms.Sequence(
    )
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

ctppsDQMRandomSource = cms.Sequence()
ctppsDQMRandomHarvest = cms.Sequence()
ctpps.toReplaceWith(ctppsDQMRandomSource, _ctppsDQMRandomSource)
ctpps.toReplaceWith(ctppsDQMRandomHarvest, _ctppsDQMRandomHarvest)

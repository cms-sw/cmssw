import FWCore.ParameterSet.Config as cms

# the following is needed for non PoolDBESSources (fake calibrations)
#
#from CalibTracker.SiStripESProducers.SiStripPedestalsFakeSource_cfi import *
from CalibTracker.SiStripESProducers.fake.SiStripQualityFakeESSource_cfi import *
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 
        'channelQuality', 
        'ZSThresholds')
)

sistripconn = cms.ESProducer("SiStripConnectivity")

# CSC trigger primitive conditions
from L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cff import *
# DT trigger primitive conditions
from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff import *
# RPC trigger conditions
#from L1TriggerConfig.RPCTriggerConfig.RPCConeConfig_cff import *
#from L1TriggerConfig.RPCTriggerConfig.RPCHwConfig_cff import *

# central L1 conditions
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskVetoAlgoTrigConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskVetoTechTrigConfig_cff import *

# end fake calibrations

from CondCore.ESSources.CondDBESSource_cfi import GlobalTag


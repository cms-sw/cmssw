import FWCore.ParameterSet.Config as cms

# the following is needed for non PoolDBESSources (fake calibrations)
#
from CalibTracker.SiStripESProducers.SiStripPedestalsFakeSource_cfi import *
from CalibTracker.SiStripESProducers.SiStripQualityFakeESSource_cfi import *
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

from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff import *
from L1TriggerConfig.RPCTriggerConfig.RPCPatSource_cfi import *
from L1TriggerConfig.RPCTriggerConfig.RPCHWConfigSource_cfi import *


# end fake calibrations

from Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi import *

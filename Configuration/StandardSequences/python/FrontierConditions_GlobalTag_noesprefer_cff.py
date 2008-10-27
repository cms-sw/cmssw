import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi import *

# the following is needed for non PoolDBESSources (fake calibrations)
#
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
from CalibTracker.SiStripESProducers.SiStripPedestalsFakeSource_cfi import *
from CalibTracker.SiStripESProducers.SiStripQualityFakeESSource_cfi import *
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *

sistripconn = cms.ESProducer("SiStripConnectivity")

hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 
        'channelQuality', 
        'ZSThresholds')
)

# L1 Fake conditions
# included for backwards compatibility with existing Global Tags
# To be removed in CMSSW 22X series
from L1Trigger.Configuration.L1Trigger_FakeConditions_cff import *

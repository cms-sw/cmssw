import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi import *

# the following is needed for non PoolDBESSources (fake calibrations)
#
from CalibMuon.DTCalibration.DTFakeVDriftESProducer_cfi import *
from CalibMuon.CSCCalibration.CSC_BadChambers_cfi import *
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
from CalibTracker.Configuration.Tracker_DependentRecords_forGlobalTag_nofakes_cff import *

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

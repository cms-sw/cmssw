import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi import *

# the following are needed for non PoolDBESSources

from CalibMuon.DTCalibration.DTFakeVDriftESProducer_cfi import *
from CalibMuon.CSCCalibration.CSC_BadChambers_cfi import *
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
from CalibTracker.Configuration.Tracker_DependentRecords_forGlobalTag_nofakes_cff import *
from CalibCalorimetry.HcalPlugins.Hcal_Conditions_forGlobalTag_cff import *
from L1Trigger.Configuration.L1Trigger_FakeConditions_cff import *
from CalibTracker.SiPixelESProducers.SiPixelFakeCPEGenericErrorParmESSource_cfi import *
from CalibTracker.SiPixelESProducers.SiPixelFakeTemplateDBObjectESSource_cfi import *

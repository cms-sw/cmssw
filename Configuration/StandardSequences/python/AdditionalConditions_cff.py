import FWCore.ParameterSet.Config as cms

# the following are needed for non PoolDBESSources
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
from CalibCalorimetry.EcalLaserCorrection.EcalLaserCorrectionServiceMC_cfi import *
from CalibCalorimetry.HcalPlugins.Hcal_Conditions_forGlobalTag_cff import *
from CalibCalorimetry.CastorCalib.CastorDbProducer_cfi import *
from CalibTracker.Configuration.Tracker_DependentRecords_forGlobalTag_nofakes_cff import *
from SimTransport.PPSProtonTransport.PPSTransportESSources_cfi import *

# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_RPC"
import FWCore.ParameterSet.Config as cms

#
#RPC calibrations
#For now, no t0 corrections are applied in the reconstruction if fake calibration is used
#(the replace should be moved to main cfg in order to avoid warning message)
from CalibMuon.RPCCalibration.RPC_Calibration_cff import *
RPCCalibPerf.connect = 'frontier://FrontierDev/CMS_COND_RPC'



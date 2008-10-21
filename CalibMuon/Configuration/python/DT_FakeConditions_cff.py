import FWCore.ParameterSet.Config as cms

#
#DT fake calibrations: es producer for t0 should introduce dependence of DTT0Rcd from geometry.
#For now, no t0 corrections are applied in the reconstruction if fake calibration is used
#(the replace should be moved to main cfg in order to avoid warning message)
from CalibMuon.DTCalibration.DTFakeTTrigESProducer_cfi import *
from CalibMuon.DTCalibration.DTFakeT0ESProducer_cfi import *
from CalibMuon.DTCalibration.DTFakeVDriftESProducer_cfi import *

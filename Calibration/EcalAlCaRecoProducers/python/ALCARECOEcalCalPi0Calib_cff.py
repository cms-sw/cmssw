import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco pi0 calibration:
#------------------------------------------------
from Calibration.EcalAlCaRecoProducers.alcastreamEcalPi0Calib_cff import *
seqALCARECOEcalCalPi0Calib = cms.Sequence(ecalpi0CalibHLT)


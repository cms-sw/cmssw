import FWCore.ParameterSet.Config as cms

#--- initialize calibration parameters for energy deposits of electrons and photons in ECAL;
#     E_calib =  pf_ECAL_calib_p0 + E_raw * pf_ECAL_calib_p1
pf_ECAL_calib_p0 = cms.double(0.0)
pf_ECAL_calib_p1 = cms.double(1.0)


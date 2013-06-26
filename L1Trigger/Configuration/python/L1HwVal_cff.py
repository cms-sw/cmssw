import FWCore.ParameterSet.Config as cms

# L1 hardware validation sequence for data
#
# run the L1 trigger emulators (including ECAL and HCAL TPG)  
#    
# V.M. Ghete 2010-02-23

from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *
L1HwVal = cms.Sequence(L1HardwareValidation)

 
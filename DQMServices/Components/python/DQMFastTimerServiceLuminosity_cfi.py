import FWCore.ParameterSet.Config as cms

# set the luminosity for the FastTimerService from SCAL
import HLTrigger.Timer.ftsLuminosityFromScalers_cfi as __ftsLuminosityFromScalers_cfi
dqmFastTimerServiceLuminosity = __ftsLuminosityFromScalers_cfi.ftsLuminosityFromScalers.clone()
dqmFastTimerServiceLuminosity.name          = 'luminosity'
dqmFastTimerServiceLuminosity.title         = 'instantaneous luminosity (from SCAL)'
dqmFastTimerServiceLuminosity.source        = 'scalersRawToDigi'
dqmFastTimerServiceLuminosity.range         = 1.e34
dqmFastTimerServiceLuminosity.resolution    = 1.e31

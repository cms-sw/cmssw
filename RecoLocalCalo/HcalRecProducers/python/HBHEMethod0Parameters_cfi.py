import FWCore.ParameterSet.Config as cms

# Behavior of "Method 0" depends on the following top-level
# HBHEPhase1Reconstructor parameters:
#
# (bool) tsFromDB
# (bool) recoParamsFromDB
#
# and on the following parameters inside the "algorithm" parameter set,
# defined as m0Parameters below:
#
# (int) firstSampleShift
# (int) samplesToAdd
# (bool) correctForPhaseContainment
# (double) correctionPhaseNS
#
# The role of "tsFromDB" parameter is to specify whether the
# "sample of interest" (SOI) should be taken from the data frame
# (in case tsFromDB is False) or from the database (tsFromDB is True).
# Once SOI is determined in this manner, the time slice for starting
# the ADC charge summation (TSS) is determined by adding SOI and
# "firstSampleShift". "firstSampleShift" can be negative, and if
# TSS = SOI + firstSampleShift ends up negative, TSS is reset to 0.
# If you want to start summation from time slice 0, just set
# "firstSampleShift" to some negative number with large magnitude.
#
# In the old reco code, it used to be possible to configure the TSS
# using a parameter value. This is no longer the case for Phase 1
# reconstructor.
#
# The values of the remaining M0 parameters, "samplesToAdd",
# "correctForPhaseContainment", and "correctionPhaseNS", will be
# taken from the configuration file only if "recoParamsFromDB"
# is set to False. If "recoParamsFromDB" is True, the config
# file values of these parameters are ignored, and these values
# are instead taken from the database. The "samplesToAdd" parameter
# defines how many contiguous time slices will be used to calculate
# the charge. TSS + samplesToAdd should not exceed the number of
# time slices in the data frame, otherwise M0 will still work but
# the results will be unreliable.
#
# Parameter "correctForPhaseContainment" specifies whether
# a correction should be made for incomplete HPD signal collection
# inside the summed time slices, and "correctionPhaseNS" specifies
# the delay (in ns) of the HPD signal w.r.t. the ADC digitization
# clock edge. That is, in the reco code, increasing correctionPhaseNS
# moves reco window to the left w.r.t. the signal (AFAIK, the behavior
# of PhaseDelay QIE11 configuration register is the opposite).
#
m0Parameters = cms.PSet(
    firstSampleShift = cms.int32(0),
    samplesToAdd = cms.int32(2),
    correctForPhaseContainment = cms.bool(True),
    correctionPhaseNS = cms.double(6.0),
)

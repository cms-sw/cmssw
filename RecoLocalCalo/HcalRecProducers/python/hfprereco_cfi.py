import FWCore.ParameterSet.Config as cms

#
# During its "normal" (i.e., non-cosmic) operation, the HFPreReconstructor
# module constructs HFPreRecHit objects using a single time slice called
# "sample of interest" (SOI). This time slice can be chosen in three
# different ways:
#    I. Take it from the data frame. This is the "standard" configuration.
#   II. Take it from the database.
#  III. Take is from a cfi parameter.
#
# HFPreReconstructor configurations corresponding to these SOI choices are:
#    I. forceSOI < 0 and tsFromDB is False.
#   II. forceSOI < 0 and tsFromDB is True.
#  III. forceSOI >= 0 (the SOI is then defined by the forceSOI value).
#
# For configuration III, the SOI value will be the same for all channels.
# For I and II, SOIs can differ from channel to channel, depending on DAQ
# or database settings, respectively.
#
# After the time slice selection outlined above, the value of parameter
# "soiShift" is added to this selection. "soiShift" can be positive,
# negative, or zero.
#
# Note that, at the time of this writing, we read out only 3 time slices
# in HF, so that meaningful non-negative values of "forceSOI" parameter
# (assuming "soiShift" value of zero) are 0, 1, and 2. In all cases (I, II,
# and III), channels for which the SOI is misconfigured will be discarded.
#
# For cosmic operation, the parameter "sumAllTimeSlices" should be set
# to "True". In this case the SOI configuration is ignored, and the energy
# is accumulated using all time slices in the data frame.
#
hfprereco = cms.EDProducer("HFPreReconstructor",
    digiLabel = cms.InputTag("hcalDigis"),
    dropZSmarkedPassed = cms.bool(False),
    tsFromDB = cms.bool(False),
    sumAllTimeSlices = cms.bool(False),
    forceSOI = cms.int32(-1),
    soiShift = cms.int32(0)
)

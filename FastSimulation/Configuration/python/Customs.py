#####################################
# a bunch of handy customisation functions
# main functions: prepareGenMixing and prepareDigiRecoMixing
# author: Lukas Vanelderen
# date:   Jan 21 2015
#####################################

import FWCore.ParameterSet.Config as cms

def disableOOTPU(process):
    process.mix.maxBunch = cms.int32(0)
    process.mix.minBunch = cms.int32(0)
    # set the bunch spacing
    # bunch spacing matters for calorimeter calibration
    # by convention bunchspace is set to 450 in case of no oot pu
    process.mix.bunchspace = 450
    return process

# more to come

#######################
# this file is the FastSim equivalent of SimGeneral/MixingModule/python/digitizers_cfi.py
# author: Lukas Vanelderen
# date:   Jan 20 2015
#######################

import FWCore.ParameterSet.Config as cms

#######################
# CONFIGURE DIGITIZERS / TRACK ACCUMULATOR / TRUTH ACCUMULATOR
#######################

from FastSimulation.Configuration.MixingModule_Full2Fast import digitizersFull2Fast
import SimGeneral.MixingModule.digitizers_cfi

theDigitizersValid = digitizersFull2Fast(SimGeneral.MixingModule.digitizers_cfi.theDigitizersValid)
theDigitizers = digitizersFull2Fast(SimGeneral.MixingModule.digitizers_cfi.theDigitizers)

#######################
# ALIASES FOR DIGI AND MIXED TRACK COLLECTIONS
#######################

# simEcalUnsuppressedDigis:   alias for ECAL digis produced by MixingModule
# simHcalUnsuppressedDigis:   alias for HCAL digis produced by MixingModule

from SimGeneral.MixingModule.digitizers_cfi import simEcalUnsuppressedDigis,simHcalUnsuppressedDigis
# alias for collections of tracks , track extras and tracker hits produced by MixingModule 
from FastSimulation.Tracking.GeneralTracksAlias_cfi import generalTracks

import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.trackCounting3D2ndComputer_cfi import *

# trackCounting3D1st btag computer
trackCounting3D1stComputer = trackCounting3D2ndComputer.clone(
    nthTrack = cms.int32(1)
)

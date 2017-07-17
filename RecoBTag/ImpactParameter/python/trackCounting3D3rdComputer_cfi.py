import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.trackCounting3D2ndComputer_cfi import *

# trackCounting3D3rd btag computer
trackCounting3D3rdComputer = trackCounting3D2ndComputer.clone(
    nthTrack = cms.int32(3)
)

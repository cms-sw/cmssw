import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.trackCounting3D2ndComputer_cfi import *

# trackCounting3D3rd btag computer
trackCounting3D3rdComputer = trackCounting3D2ndComputer.clone(
    nthTrack = 3
)
# foo bar baz
# TQLU3U7AfbvXj

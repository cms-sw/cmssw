import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.negativeTrackCounting3D2ndComputer_cfi import *

# negativeTrackCounting3D3rd btag computer
negativeTrackCounting3D3rdComputer = negativeTrackCounting3D2ndComputer.clone(
    nthTrack = cms.int32(3)
)



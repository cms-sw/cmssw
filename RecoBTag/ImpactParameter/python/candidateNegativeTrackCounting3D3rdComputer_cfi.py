import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.candidateNegativeTrackCounting3D2ndComputer_cfi import *

# negativeTrackCounting3D3rd btag computer
candidateNegativeTrackCounting3D3rdComputer = candidateNegativeTrackCounting3D2ndComputer.clone(
    nthTrack = cms.int32(3)
)

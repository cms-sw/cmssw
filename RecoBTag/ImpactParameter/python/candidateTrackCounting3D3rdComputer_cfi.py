import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.candidateTrackCounting3D2ndComputer_cfi import *

# trackCounting3D3rd btag computer
candidateTrackCounting3D3rdComputer = candidateTrackCounting3D2ndComputer.clone(
    nthTrack = cms.int32(3)
)

"""
_Utils_
Tools to customise the DQM offline configuration run on the dedicated express-like stream during pp_on_AA
"""

import FWCore.ParameterSet.Config as cms

def customise_PPonAATrackingOnlyDQM(process):
    if hasattr(process,'dqmofflineOnPAT_step') or hasattr(process,'dqmoffline_step'):
        process=customise_DQMSequenceHiConformalTracks(process)
    return process

def customise_DQMSequenceHiConformalTracks(process):
    process.TrackingDQMSourceTier0Common.remove(process.hiConformalPixelTracksQA)
    process.TrackingDQMSourceTier0MinBias.remove(process.hiConformalPixelTracksQA)
    process.TrackingDQMSourceTier0.remove(process.hiConformalPixelTracksQA)
    return process



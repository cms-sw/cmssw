import FWCore.ParameterSet.Config as cms

from ..modules.TTTrackAssociatorFromPixelDigis_cfi import *

TrackTriggerAssociatorTracks = cms.Sequence(TTTrackAssociatorFromPixelDigis)

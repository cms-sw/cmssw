import FWCore.ParameterSet.Config as cms

from ..modules.TTTrackAssociatorFromPixelDigisExtended_cfi import *
from ..sequences.TrackTriggerAssociatorTracks_cfi import *
from ..modules.l1tTTTracksFromExtendedTrackletEmulation_cfi import *
from ..modules.l1tTTTracksFromTrackletEmulation_cfi import *
from ..modules.offlineBeamSpot_cfi import *

L1TPromptExtendedHybridTracksWithAssociators = cms.Sequence(offlineBeamSpot+l1tTTTracksFromTrackletEmulation+TrackTriggerAssociatorTracks+l1tTTTracksFromExtendedTrackletEmulation+TTTrackAssociatorFromPixelDigisExtended)

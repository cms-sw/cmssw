import FWCore.ParameterSet.Config as cms

from ..modules.hltGeneralTracks_cfi import *
from ..modules.hltTrackerClusterCheck_cfi import *
from ..sequences.HLTHighPtTripletStepSequence_cfi import *
from ..sequences.HLTPhase2PixelTracksAndVerticesSequence_cfi import *
from ..sequences.HLTInitialStepSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *

HLTTrackingSequence = cms.Sequence(HLTItLocalRecoSequence+
                                   HLTOtLocalRecoSequence+
                                   hltTrackerClusterCheck+
                                   HLTPhase2PixelTracksAndVerticesSequence+
                                   HLTInitialStepSequence+
                                   HLTHighPtTripletStepSequence+
                                   hltGeneralTracks)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
singleIterPatatrack.toReplaceWith(HLTTrackingSequence, HLTTrackingSequence.copyAndExclude([HLTHighPtTripletStepSequence]))

from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
(ngtScouting & trackingLST).toReplaceWith(HLTTrackingSequence, HLTTrackingSequence.copyAndExclude([HLTHighPtTripletStepSequence]))
(ngtScouting & ~trackingLST).toReplaceWith(HLTTrackingSequence, HLTTrackingSequence.copyAndExclude([HLTInitialStepSequence,HLTHighPtTripletStepSequence]))

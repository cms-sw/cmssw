import FWCore.ParameterSet.Config as cms

from ..modules.hltHighPtTripletStepTrackCandidates_cfi import *
from ..modules.hltHighPtTripletStepTrackCutClassifier_cfi import *
from ..modules.hltHighPtTripletStepTracks_cfi import *
from ..modules.hltHighPtTripletStepTrackSelectionHighPurity_cfi import *
from ..sequences.HLTHighPtTripletStepSeedingSequence_cfi import *

HLTHighPtTripletStepSequence = cms.Sequence(
     HLTHighPtTripletStepSeedingSequence
    +hltHighPtTripletStepTrackCandidates
    +hltHighPtTripletStepTracks
    +hltHighPtTripletStepTrackCutClassifier
    +hltHighPtTripletStepTrackSelectionHighPurity
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(HLTHighPtTripletStepSequence, HLTHighPtTripletStepSequence.copyAndExclude([HLTHighPtTripletStepSeedingSequence]))

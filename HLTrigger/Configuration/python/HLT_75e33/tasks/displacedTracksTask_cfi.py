import FWCore.ParameterSet.Config as cms

from ..modules.displacedTracks_cfi import *
from ..modules.duplicateDisplacedTrackCandidates_cfi import *
from ..modules.duplicateDisplacedTrackClassifier_cfi import *
from ..modules.mergedDuplicateDisplacedTracks_cfi import *

displacedTracksTask = cms.Task(displacedTracks, duplicateDisplacedTrackCandidates, duplicateDisplacedTrackClassifier, mergedDuplicateDisplacedTracks)

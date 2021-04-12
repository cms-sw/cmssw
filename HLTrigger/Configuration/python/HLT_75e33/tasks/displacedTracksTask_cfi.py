import FWCore.ParameterSet.Config as cms

from ..modules.duplicateDisplacedTrackCandidates_cfi import *
from ..modules.duplicateDisplacedTrackClassifier_cfi import *
from ..modules.mergedDuplicateDisplacedTracks_cfi import *

displacedTracksTask = cms.Task(duplicateDisplacedTrackCandidates, duplicateDisplacedTrackClassifier, mergedDuplicateDisplacedTracks)

import FWCore.ParameterSet.Config as cms

from ..modules.me0RecHits_cfi import *
from ..modules.me0Segments_cfi import *

me0LocalRecoTask = cms.Task(
    me0RecHits,
    me0Segments
)

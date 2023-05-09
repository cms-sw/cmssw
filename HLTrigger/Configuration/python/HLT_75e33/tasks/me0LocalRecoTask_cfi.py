import FWCore.ParameterSet.Config as cms

from ..modules.hltMe0RecHits_cfi import *
from ..modules.hltMe0Segments_cfi import *

me0LocalRecoTask = cms.Task(
    hltMe0RecHits,
    hltMe0Segments
)

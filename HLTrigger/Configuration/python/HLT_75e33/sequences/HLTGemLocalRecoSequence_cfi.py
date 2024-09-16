import FWCore.ParameterSet.Config as cms

from ..modules.hltGemRecHits_cfi import *
from ..modules.hltGemSegments_cfi import *

HLTGemLocalRecoSequence = cms.Sequence(hltGemRecHits+hltGemSegments)

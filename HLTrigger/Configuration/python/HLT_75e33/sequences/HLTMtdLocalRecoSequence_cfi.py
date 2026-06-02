import FWCore.ParameterSet.Config as cms

from ..modules.hltMtdUncalibratedRecHits_cfi import *
from ..modules.hltMtdRecHits_cfi import *
from ..modules.hltMtdTrackingRecHits_cfi import *
from ..modules.hltMtdClusters_cfi import *

HLTMtdLocalRecoSequence = cms.Sequence(hltMtdUncalibratedRecHits
                                       +hltMtdRecHits
                                       +hltMtdClusters
                                       +hltMtdTrackingRecHits)

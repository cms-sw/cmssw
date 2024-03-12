import FWCore.ParameterSet.Config as cms

from ..modules.siPixelClusters_cfi import *
from ..modules.siPixelRecHits_cfi import *

HLTDoLocalPixelTask = cms.Task(
    siPixelClusters,
    siPixelRecHits
)
# foo bar baz
# u1TS8IqVI4MjX
# YQ6h0PqKcxqPP

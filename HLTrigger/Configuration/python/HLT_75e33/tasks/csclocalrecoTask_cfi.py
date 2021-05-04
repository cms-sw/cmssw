import FWCore.ParameterSet.Config as cms

from ..modules.csc2DRecHits_cfi import *
from ..modules.cscSegments_cfi import *

csclocalrecoTask = cms.Task(
    csc2DRecHits,
    cscSegments
)

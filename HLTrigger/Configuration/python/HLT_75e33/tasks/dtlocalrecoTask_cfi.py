import FWCore.ParameterSet.Config as cms

from ..modules.dt1DCosmicRecHits_cfi import *
from ..modules.dt1DRecHits_cfi import *
from ..modules.dt4DCosmicSegments_cfi import *
from ..modules.dt4DSegments_cfi import *

dtlocalrecoTask = cms.Task(dt1DCosmicRecHits, dt1DRecHits, dt4DCosmicSegments, dt4DSegments)

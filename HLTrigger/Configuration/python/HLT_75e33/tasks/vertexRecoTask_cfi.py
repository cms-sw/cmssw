import FWCore.ParameterSet.Config as cms

from ..modules.ak4CaloJetsForTrk_cfi import *
from ..modules.offlinePrimaryVertices_cfi import *
from ..modules.trackRefsForJetsBeforeSorting_cfi import *
from ..modules.trackWithVertexRefSelectorBeforeSorting_cfi import *
from ..modules.unsortedOfflinePrimaryVertices_cfi import *
from ..tasks.initialStepPVTask_cfi import *

vertexRecoTask = cms.Task(
    ak4CaloJetsForTrk,
    initialStepPVTask,
    offlinePrimaryVertices,
    trackRefsForJetsBeforeSorting,
    trackWithVertexRefSelectorBeforeSorting,
    unsortedOfflinePrimaryVertices
)

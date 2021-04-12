import FWCore.ParameterSet.Config as cms

from ..modules.ak4CaloJetsForTrk_cfi import *
from ..modules.inclusiveSecondaryVertices_cfi import *
from ..modules.inclusiveVertexFinder_cfi import *
from ..modules.offlinePrimaryVertices_cfi import *
from ..modules.offlinePrimaryVerticesWithBS_cfi import *
from ..modules.trackRefsForJetsBeforeSorting_cfi import *
from ..modules.trackVertexArbitrator_cfi import *
from ..modules.trackWithVertexRefSelectorBeforeSorting_cfi import *
from ..modules.unsortedOfflinePrimaryVertices_cfi import *
from ..modules.vertexMerger_cfi import *
from ..tasks.initialStepPVTask_cfi import *

vertexRecoTask = cms.Task(ak4CaloJetsForTrk, inclusiveSecondaryVertices, inclusiveVertexFinder, initialStepPVTask, offlinePrimaryVertices, offlinePrimaryVerticesWithBS, trackRefsForJetsBeforeSorting, trackVertexArbitrator, trackWithVertexRefSelectorBeforeSorting, unsortedOfflinePrimaryVertices, vertexMerger)

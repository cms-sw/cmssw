import FWCore.ParameterSet.Config as cms

from ..modules.offlinePrimaryVertices_cfi import *
from ..modules.trackRefsForJetsBeforeSorting_cfi import *
from ..modules.trackWithVertexRefSelectorBeforeSorting_cfi import *
from ..modules.unsortedOfflinePrimaryVertices_cfi import *
from ..sequences.initialStepPVSequence_cfi import *

vertexRecoSequence = cms.Sequence(initialStepPVSequence+unsortedOfflinePrimaryVertices+trackWithVertexRefSelectorBeforeSorting+trackRefsForJetsBeforeSorting+offlinePrimaryVertices)

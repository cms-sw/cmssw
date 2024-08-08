import FWCore.ParameterSet.Config as cms

from ..modules.hltOfflinePrimaryVertices_cfi import *
from ..modules.hltTrackRefsForJetsBeforeSorting_cfi import *
from ..modules.hltTrackWithVertexRefSelectorBeforeSorting_cfi import *
from ..modules.hltUnsortedOfflinePrimaryVertices_cfi import *
from ..sequences.HLTInitialStepPVSequence_cfi import *

HLTVertexRecoSequence = cms.Sequence(HLTInitialStepPVSequence+hltUnsortedOfflinePrimaryVertices+hltTrackWithVertexRefSelectorBeforeSorting+hltTrackRefsForJetsBeforeSorting+hltOfflinePrimaryVertices)

import FWCore.ParameterSet.Config as cms

from ..modules.hltOfflinePrimaryVertices_cfi import *
from ..modules.hltTrackRefsForJetsBeforeSorting_cfi import *
from ..modules.hltTrackWithVertexRefSelectorBeforeSorting_cfi import *
from ..modules.hltUnsortedOfflinePrimaryVertices_cfi import *
from ..sequences.HLTInitialStepPVSequence_cfi import *
from ..sequences.HLTVertex4DRecoSequence_cfi import *

HLTVertexRecoSequence = cms.Sequence(HLTInitialStepPVSequence+
                                     hltUnsortedOfflinePrimaryVertices+
                                     hltTrackWithVertexRefSelectorBeforeSorting+
                                     hltTrackRefsForJetsBeforeSorting+
                                     hltOfflinePrimaryVertices)

_HLTVertexRecoSequence = HLTVertexRecoSequence.copy()

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
mtd_at_hlt.toReplaceWith(HLTVertexRecoSequence,
                         cms.Sequence(_HLTVertexRecoSequence+
                                      HLTVertex4DRecoSequence))

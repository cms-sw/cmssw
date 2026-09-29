import FWCore.ParameterSet.Config as cms

from ..modules.hltTofPID4DnoPID_cfi import *
from ..modules.hltUnsortedOfflinePrimaryVertices4D_cfi import *
from ..modules.hltTrackWithVertexRefSelectorBeforeSorting4D_cfi import *
from ..modules.hltTrackRefsForJetsBeforeSorting4D_cfi import *
from ..modules.hltOfflinePrimaryVertices4D_cfi import *
from ..modules.hltTofPID_cfi import *

HLTVertex4DRecoSequence = cms.Sequence(hltTofPID4DnoPID+
                                       hltUnsortedOfflinePrimaryVertices4D+
                                       hltTrackWithVertexRefSelectorBeforeSorting4D+
                                       hltTrackRefsForJetsBeforeSorting4D+
                                       hltOfflinePrimaryVertices4D+
                                       hltTofPID)

import FWCore.ParameterSet.Config as cms

from ..modules.hltTrackExtenderWithMTD_cfi import *
from ..modules.hltMtdTrackQualityMVA_cfi import *

HLTFastTimingGlobalRecoSequence = cms.Sequence(hltTrackExtenderWithMTD
                                               +hltMtdTrackQualityMVA
                                               )

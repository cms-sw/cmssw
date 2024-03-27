import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6L1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEG32EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForSingleEleIsolatedFilter_cfi import *
from ..modules.hltEle32WPTightBestGsfChi2L1SeededFilter_cfi import *
from ..modules.hltEle32WPTightBestGsfNLayerITL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightEcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfDetaL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfDphiL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfTrackIsoL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightHcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightHEL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightHgcalHEL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightHgcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightPixelMatchL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightPMS2L1SeededFilter_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *

HLTEle32WPTightL1SeededInnerSequence = cms.Sequence(hltEgammaHGCALIDVarsL1Seeded
    +hltEgammaEcalPFClusterIsoL1Seeded
    +hltEgammaHGCalLayerClusterIsoL1Seeded
    +hltEgammaHcalPFClusterIsoL1Seeded
    +hltEgammaEleL1TrkIsoL1Seeded
    +hltEgammaCandidatesWrapperL1Seeded
    +hltEG32EtL1SeededFilter
    +hltEle32WPTightClusterShapeL1SeededFilter
    +hltEle32WPTightClusterShapeSigmavvL1SeededFilter
    +hltEle32WPTightClusterShapeSigmawwL1SeededFilter
    +hltEle32WPTightHgcalHEL1SeededFilter
    +hltEle32WPTightHEL1SeededFilter
    +hltEle32WPTightEcalIsoL1SeededFilter
    +hltEle32WPTightHgcalIsoL1SeededFilter
    +hltEle32WPTightHcalIsoL1SeededFilter
    +hltEle32WPTightPixelMatchL1SeededFilter
    +hltEle32WPTightPMS2L1SeededFilter
    +hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter
    +hltEle32WPTightGsfDetaL1SeededFilter
    +hltEle32WPTightGsfDphiL1SeededFilter
    +hltEle32WPTightBestGsfNLayerITL1SeededFilter
    +hltEle32WPTightBestGsfChi2L1SeededFilter
    +hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter
    +HLTTrackingV61Sequence
    +hltEgammaEleGsfTrackIsoV6L1Seeded
    +hltEle32WPTightGsfTrackIsoL1SeededFilter)

import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6Unseeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *
from ..modules.hltEG32EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEGL1SeedsForSingleEleIsolatedFilter_cfi import *
from ..modules.hltEle32WPTightBestGsfChi2UnseededFilter_cfi import *
from ..modules.hltEle32WPTightBestGsfNLayerITUnseededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeSigmavvUnseededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeSigmawwUnseededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeUnseededFilter_cfi import *
from ..modules.hltEle32WPTightEcalIsoUnseededFilter_cfi import *
from ..modules.hltEle32WPTightGsfDetaUnseededFilter_cfi import *
from ..modules.hltEle32WPTightGsfDphiUnseededFilter_cfi import *
from ..modules.hltEle32WPTightGsfOneOEMinusOneOPUnseededFilter_cfi import *
from ..modules.hltEle32WPTightGsfTrackIsoFromL1TracksUnseededFilter_cfi import *
from ..modules.hltEle32WPTightGsfTrackIsoUnseededFilter_cfi import *
from ..modules.hltEle32WPTightHcalIsoUnseededFilter_cfi import *
from ..modules.hltEle32WPTightHEUnseededFilter_cfi import *
from ..modules.hltEle32WPTightHgcalHEUnseededFilter_cfi import *
from ..modules.hltEle32WPTightHgcalIsoUnseededFilter_cfi import *
from ..modules.hltEle32WPTightPixelMatchUnseededFilter_cfi import *
from ..modules.hltEle32WPTightPMS2UnseededFilter_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *

HLTEle32WPTightUnseededInnerSequence = cms.Sequence(hltEgammaHGCALIDVarsUnseeded
    +hltEgammaEcalPFClusterIsoUnseeded
    +hltEgammaHGCalLayerClusterIsoUnseeded
    +hltEgammaHcalPFClusterIsoUnseeded
    +hltEgammaEleL1TrkIsoUnseeded
    +hltEgammaHoverEUnseeded
    +hltEgammaCandidatesUnseeded
    +hltEgammaClusterShapeUnseeded
    +hltEgammaCandidatesWrapperUnseeded
    +hltEG32EtUnseededFilter
    +hltEle32WPTightClusterShapeUnseededFilter
    +hltEle32WPTightClusterShapeSigmavvUnseededFilter
    +hltEle32WPTightClusterShapeSigmawwUnseededFilter
    +hltEle32WPTightHgcalHEUnseededFilter
    +hltEle32WPTightHEUnseededFilter
    +hltEle32WPTightEcalIsoUnseededFilter
    +hltEle32WPTightHgcalIsoUnseededFilter
    +hltEle32WPTightHcalIsoUnseededFilter
    +hltEle32WPTightPixelMatchUnseededFilter
    +hltEle32WPTightPMS2UnseededFilter
    +hltEle32WPTightGsfOneOEMinusOneOPUnseededFilter
    +hltEle32WPTightGsfDetaUnseededFilter
    +hltEle32WPTightGsfDphiUnseededFilter
    +hltEle32WPTightBestGsfNLayerITUnseededFilter
    +hltEle32WPTightBestGsfChi2UnseededFilter
    +hltEle32WPTightGsfTrackIsoFromL1TracksUnseededFilter
    +HLTTrackingV61Sequence
    +hltEgammaEleGsfTrackIsoV6Unseeded
    +hltEle32WPTightGsfTrackIsoUnseededFilter
)

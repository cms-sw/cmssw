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
from ..modules.hltEG26EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEGL1SeedsForSingleEleIsolatedFilter_cfi import *
from ..modules.hltEle26WP70BestGsfChi2UnseededFilter_cfi import *
from ..modules.hltEle26WP70BestGsfNLayerITUnseededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeSigmavvUnseededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeSigmawwUnseededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeUnseededFilter_cfi import *
from ..modules.hltEle26WP70EcalIsoUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfDetaUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfDphiUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfOneOEMinusOneOPUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfTrackIsoFromL1TracksUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfTrackIsoUnseededFilter_cfi import *
from ..modules.hltEle26WP70HcalIsoUnseededFilter_cfi import *
from ..modules.hltEle26WP70HEUnseededFilter_cfi import *
from ..modules.hltEle26WP70HgcalHEUnseededFilter_cfi import *
from ..modules.hltEle26WP70HgcalIsoUnseededFilter_cfi import *
from ..modules.hltEle26WP70PixelMatchUnseededFilter_cfi import *
from ..modules.hltEle26WP70PMS2UnseededFilter_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *

HLTEle26WP70UnseededInnerSequence = cms.Sequence(hltEgammaCandidatesUnseeded
    +hltEgammaClusterShapeUnseeded
    +hltEgammaHGCALIDVarsUnseeded
    +hltEgammaHoverEUnseeded
    +hltEgammaEcalPFClusterIsoUnseeded
    +hltEgammaHGCalLayerClusterIsoUnseeded
    +hltEgammaHcalPFClusterIsoUnseeded
    +hltEgammaEleL1TrkIsoUnseeded
    +hltEgammaCandidatesWrapperUnseeded
    +hltEG26EtUnseededFilter
    +hltEle26WP70ClusterShapeUnseededFilter
    +hltEle26WP70ClusterShapeSigmavvUnseededFilter
    +hltEle26WP70ClusterShapeSigmawwUnseededFilter
    +hltEle26WP70HgcalHEUnseededFilter
    +hltEle26WP70HEUnseededFilter
    +hltEle26WP70EcalIsoUnseededFilter
    +hltEle26WP70HgcalIsoUnseededFilter
    +hltEle26WP70HcalIsoUnseededFilter
    +hltEle26WP70PixelMatchUnseededFilter
    +hltEle26WP70PMS2UnseededFilter
    +hltEle26WP70GsfOneOEMinusOneOPUnseededFilter
    +hltEle26WP70GsfDetaUnseededFilter
    +hltEle26WP70GsfDphiUnseededFilter
    +hltEle26WP70BestGsfNLayerITUnseededFilter
    +hltEle26WP70BestGsfChi2UnseededFilter
    +hltEle26WP70GsfTrackIsoFromL1TracksUnseededFilter
    +HLTTrackingV61Sequence
    +hltEgammaEleGsfTrackIsoV6Unseeded
    +hltEle26WP70GsfTrackIsoUnseededFilter
    )

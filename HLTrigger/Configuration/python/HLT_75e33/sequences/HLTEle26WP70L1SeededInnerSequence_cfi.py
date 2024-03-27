import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6L1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *
from ..modules.hltEG26EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForSingleEleIsolatedFilter_cfi import *
from ..modules.hltEle26WP70BestGsfChi2L1SeededFilter_cfi import *
from ..modules.hltEle26WP70BestGsfNLayerITL1SeededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeL1SeededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltEle26WP70EcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle26WP70GsfDetaL1SeededFilter_cfi import *
from ..modules.hltEle26WP70GsfDphiL1SeededFilter_cfi import *
from ..modules.hltEle26WP70GsfOneOEMinusOneOPL1SeededFilter_cfi import *
from ..modules.hltEle26WP70GsfTrackIsoFromL1TracksL1SeededFilter_cfi import *
from ..modules.hltEle26WP70GsfTrackIsoL1SeededFilter_cfi import *
from ..modules.hltEle26WP70HcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle26WP70HEL1SeededFilter_cfi import *
from ..modules.hltEle26WP70HgcalHEL1SeededFilter_cfi import *
from ..modules.hltEle26WP70HgcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle26WP70PixelMatchL1SeededFilter_cfi import *
from ..modules.hltEle26WP70PMS2L1SeededFilter_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *

HLTEle26WP70L1SeededInnerSequence = cms.Sequence(hltEgammaCandidatesL1Seeded
    +hltEgammaClusterShapeL1Seeded
    +hltEgammaHGCALIDVarsL1Seeded
    +hltEgammaHoverEL1Seeded
    +hltEgammaEcalPFClusterIsoL1Seeded
    +hltEgammaHGCalLayerClusterIsoL1Seeded
    +hltEgammaHcalPFClusterIsoL1Seeded
    +hltEgammaEleL1TrkIsoL1Seeded
    +hltEgammaCandidatesWrapperL1Seeded
    +hltEG26EtL1SeededFilter
    +hltEle26WP70ClusterShapeL1SeededFilter
    +hltEle26WP70ClusterShapeSigmavvL1SeededFilter
    +hltEle26WP70ClusterShapeSigmawwL1SeededFilter
    +hltEle26WP70HgcalHEL1SeededFilter
    +hltEle26WP70HEL1SeededFilter
    +hltEle26WP70EcalIsoL1SeededFilter
    +hltEle26WP70HgcalIsoL1SeededFilter
    +hltEle26WP70HcalIsoL1SeededFilter
    +hltEle26WP70PixelMatchL1SeededFilter
    +hltEle26WP70PMS2L1SeededFilter
    +hltEle26WP70GsfOneOEMinusOneOPL1SeededFilter
    +hltEle26WP70GsfDetaL1SeededFilter
    +hltEle26WP70GsfDphiL1SeededFilter
    +hltEle26WP70BestGsfNLayerITL1SeededFilter
    +hltEle26WP70BestGsfChi2L1SeededFilter
    +hltEle26WP70GsfTrackIsoFromL1TracksL1SeededFilter
    +HLTTrackingV61Sequence
    +hltEgammaEleGsfTrackIsoV6L1Seeded
    +hltEle26WP70GsfTrackIsoL1SeededFilter
    )

import FWCore.ParameterSet.Config as cms

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
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchUnseededSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTGsfElectronUnseededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..tasks.HLTEle26WP70UnseededTask_cfi import *

HLTEle26WP70UnseededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForSingleEleIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltEG26EtUnseededFilter +
    hltEle26WP70ClusterShapeUnseededFilter +
    hltEle26WP70ClusterShapeSigmavvUnseededFilter +
    hltEle26WP70ClusterShapeSigmawwUnseededFilter +
    hltEle26WP70HgcalHEUnseededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltEle26WP70HEUnseededFilter +
    hltEle26WP70EcalIsoUnseededFilter +
    hltEle26WP70HgcalIsoUnseededFilter +
    HLTPFHcalClusteringForEgamma +
    hltEle26WP70HcalIsoUnseededFilter +
    HLTElePixelMatchUnseededSequence +
    hltEle26WP70PixelMatchUnseededFilter +
    hltEle26WP70PMS2UnseededFilter +
    HLTGsfElectronUnseededSequence +
    hltEle26WP70GsfOneOEMinusOneOPUnseededFilter +
    hltEle26WP70GsfDetaUnseededFilter +
    hltEle26WP70GsfDphiUnseededFilter +
    hltEle26WP70BestGsfNLayerITUnseededFilter +
    hltEle26WP70BestGsfChi2UnseededFilter +
    hltEle26WP70GsfTrackIsoFromL1TracksUnseededFilter +
    HLTTrackingV61Sequence +
    hltEle26WP70GsfTrackIsoUnseededFilter,
    HLTEle26WP70UnseededTask
)

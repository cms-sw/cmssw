import FWCore.ParameterSet.Config as cms

from ..modules.hltEG5EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEle5DphiUnseededFilter_cfi import *
from ..modules.hltEle5WP70BestGsfChi2UnseededFilter_cfi import *
from ..modules.hltEle5WP70BestGsfNLayerITUnseededFilter_cfi import *
from ..modules.hltEle5WP70ClusterShapeSigmavvUnseededFilter_cfi import *
from ..modules.hltEle5WP70ClusterShapeSigmawwUnseededFilter_cfi import *
from ..modules.hltEle5WP70ClusterShapeUnseededFilter_cfi import *
from ..modules.hltEle5WP70EcalIsoUnseededFilter_cfi import *
from ..modules.hltEle5WP70GsfDetaUnseededFilter_cfi import *
from ..modules.hltEle5WP70GsfDphiUnseededFilter_cfi import *
from ..modules.hltEle5WP70GsfOneOEMinusOneOPUnseededFilter_cfi import *
from ..modules.hltEle5WP70GsfTrackIsoFromL1TracksUnseededFilter_cfi import *
from ..modules.hltEle5WP70GsfTrackIsoUnseededFilter_cfi import *
from ..modules.hltEle5WP70HcalIsoUnseededFilter_cfi import *
from ..modules.hltEle5WP70HEUnseededFilter_cfi import *
from ..modules.hltEle5WP70HgcalHEUnseededFilter_cfi import *
from ..modules.hltEle5WP70HgcalIsoUnseededFilter_cfi import *
from ..modules.hltEle5WP70PixelMatchUnseededFilter_cfi import *
from ..modules.hltEle5WP70PMS2UnseededFilter_cfi import *
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
from ..tasks.HLTEle5WP70OpenUnseededTask_cfi import *

HLTEle5WP70OpenUnseededSequence = cms.Sequence(
    HLTL1Sequence +
    HLTDoFullUnpackingEgammaEcalSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltEG5EtUnseededFilter +
    cms.ignore(hltEle5WP70ClusterShapeUnseededFilter) +
    cms.ignore(hltEle5WP70ClusterShapeSigmavvUnseededFilter) +
    cms.ignore(hltEle5WP70ClusterShapeSigmawwUnseededFilter) +
    cms.ignore(hltEle5WP70HgcalHEUnseededFilter) +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    cms.ignore(hltEle5WP70HEUnseededFilter) +
    cms.ignore(hltEle5WP70EcalIsoUnseededFilter) +
    cms.ignore(hltEle5WP70HgcalIsoUnseededFilter) +
    HLTPFHcalClusteringForEgamma +
    cms.ignore(hltEle5WP70HcalIsoUnseededFilter) +
    HLTElePixelMatchUnseededSequence +
    cms.ignore(hltEle5WP70PixelMatchUnseededFilter) +
    cms.ignore(hltEle5WP70PMS2UnseededFilter) +
    HLTGsfElectronUnseededSequence +
    cms.ignore(hltEle5WP70GsfOneOEMinusOneOPUnseededFilter) +
    cms.ignore(hltEle5WP70GsfDetaUnseededFilter) +
    cms.ignore(hltEle5WP70GsfDphiUnseededFilter) +
    cms.ignore(hltEle5WP70BestGsfNLayerITUnseededFilter) +
    cms.ignore(hltEle5WP70BestGsfChi2UnseededFilter) +
    hltEle5DphiUnseededFilter +
    cms.ignore(hltEle5WP70GsfTrackIsoFromL1TracksUnseededFilter) +
    HLTTrackingV61Sequence +
    cms.ignore(hltEle5WP70GsfTrackIsoUnseededFilter),
    HLTEle5WP70OpenUnseededTask
)

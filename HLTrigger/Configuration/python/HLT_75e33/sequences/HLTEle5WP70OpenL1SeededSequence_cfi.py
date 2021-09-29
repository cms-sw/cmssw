import FWCore.ParameterSet.Config as cms

from ..modules.hltEG5EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEle5DphiL1SeededFilter_cfi import *
from ..modules.hltEle5WP70BestGsfChi2L1SeededFilter_cfi import *
from ..modules.hltEle5WP70BestGsfNLayerITL1SeededFilter_cfi import *
from ..modules.hltEle5WP70ClusterShapeL1SeededFilter_cfi import *
from ..modules.hltEle5WP70ClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltEle5WP70ClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltEle5WP70EcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle5WP70GsfDetaL1SeededFilter_cfi import *
from ..modules.hltEle5WP70GsfDphiL1SeededFilter_cfi import *
from ..modules.hltEle5WP70GsfOneOEMinusOneOPL1SeededFilter_cfi import *
from ..modules.hltEle5WP70GsfTrackIsoFromL1TracksL1SeededFilter_cfi import *
from ..modules.hltEle5WP70GsfTrackIsoL1SeededFilter_cfi import *
from ..modules.hltEle5WP70HcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle5WP70HEL1SeededFilter_cfi import *
from ..modules.hltEle5WP70HgcalHEL1SeededFilter_cfi import *
from ..modules.hltEle5WP70HgcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle5WP70PixelMatchL1SeededFilter_cfi import *
from ..modules.hltEle5WP70PMS2L1SeededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTGsfElectronL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..tasks.HLTEle5WP70OpenL1SeededTask_cfi import *

HLTEle5WP70OpenL1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG5EtL1SeededFilter +
    hltEle5WP70ClusterShapeL1SeededFilter +
    hltEle5WP70ClusterShapeSigmavvL1SeededFilter +
    hltEle5WP70ClusterShapeSigmawwL1SeededFilter +
    hltEle5WP70HgcalHEL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltEle5WP70HEL1SeededFilter +
    hltEle5WP70EcalIsoL1SeededFilter +
    hltEle5WP70HgcalIsoL1SeededFilter +
    HLTPFHcalClusteringForEgamma +
    hltEle5WP70HcalIsoL1SeededFilter +
    HLTElePixelMatchL1SeededSequence +
    hltEle5WP70PixelMatchL1SeededFilter +
    hltEle5WP70PMS2L1SeededFilter +
    HLTGsfElectronL1SeededSequence +
    hltEle5WP70GsfOneOEMinusOneOPL1SeededFilter +
    hltEle5WP70GsfDetaL1SeededFilter +
    hltEle5WP70GsfDphiL1SeededFilter +
    hltEle5WP70BestGsfNLayerITL1SeededFilter +
    hltEle5WP70BestGsfChi2L1SeededFilter +
    hltEle5DphiL1SeededFilter +
    hltEle5WP70GsfTrackIsoFromL1TracksL1SeededFilter +
    HLTTrackingV61Sequence +
    hltEle5WP70GsfTrackIsoL1SeededFilter,
    HLTEle5WP70OpenL1SeededTask
)

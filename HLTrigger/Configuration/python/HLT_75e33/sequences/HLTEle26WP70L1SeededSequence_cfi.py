import FWCore.ParameterSet.Config as cms

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
from ..tasks.HLTEle26WP70L1SeededTask_cfi import *

HLTEle26WP70L1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForSingleEleIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG26EtL1SeededFilter +
    hltEle26WP70ClusterShapeL1SeededFilter +
    hltEle26WP70ClusterShapeSigmavvL1SeededFilter +
    hltEle26WP70ClusterShapeSigmawwL1SeededFilter +
    hltEle26WP70HgcalHEL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltEle26WP70HEL1SeededFilter +
    hltEle26WP70EcalIsoL1SeededFilter +
    hltEle26WP70HgcalIsoL1SeededFilter +
    HLTPFHcalClusteringForEgamma +
    hltEle26WP70HcalIsoL1SeededFilter +
    HLTElePixelMatchL1SeededSequence +
    hltEle26WP70PixelMatchL1SeededFilter +
    hltEle26WP70PMS2L1SeededFilter +
    HLTGsfElectronL1SeededSequence +
    hltEle26WP70GsfOneOEMinusOneOPL1SeededFilter +
    hltEle26WP70GsfDetaL1SeededFilter +
    hltEle26WP70GsfDphiL1SeededFilter +
    hltEle26WP70BestGsfNLayerITL1SeededFilter +
    hltEle26WP70BestGsfChi2L1SeededFilter +
    hltEle26WP70GsfTrackIsoFromL1TracksL1SeededFilter +
    HLTTrackingV61Sequence +
    hltEle26WP70GsfTrackIsoL1SeededFilter,
    HLTEle26WP70L1SeededTask
)

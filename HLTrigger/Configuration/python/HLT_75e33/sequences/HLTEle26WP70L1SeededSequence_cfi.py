import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTGsfElectronL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *

from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoL1Seeded_cfi import *
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

HLTEle26WP70L1SeededSequence = cms.Sequence(
    hltEGL1SeedsForSingleEleIsolatedFilter
    +HLTDoFullUnpackingEgammaEcalL1SeededSequence
    +HLTPFClusteringForEgammaL1SeededSequence
    +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
    +hltEgammaCandidatesL1Seeded
    +hltEgammaCandidatesWrapperL1Seeded
    +hltEG26EtL1SeededFilter
    +hltEgammaClusterShapeL1Seeded
    +hltEle26WP70ClusterShapeL1SeededFilter
    +hltEgammaHGCALIDVarsL1Seeded
    +hltEle26WP70ClusterShapeSigmavvL1SeededFilter
    +hltEle26WP70ClusterShapeSigmawwL1SeededFilter
    +hltEle26WP70HgcalHEL1SeededFilter
    +HLTEGammaDoLocalHcalSequence
    +HLTFastJetForEgammaSequence
    +hltEgammaHoverEL1Seeded
    +hltEle26WP70HEL1SeededFilter
    +hltEgammaEcalPFClusterIsoL1Seeded
    +hltEle26WP70EcalIsoL1SeededFilter
    +hltEgammaHGCalLayerClusterIsoL1Seeded
    +hltEle26WP70HgcalIsoL1SeededFilter
    +HLTPFHcalClusteringForEgammaSequence
    +hltEgammaHcalPFClusterIsoL1Seeded
    +hltEle26WP70HcalIsoL1SeededFilter
    +HLTElePixelMatchL1SeededSequence
    +hltEle26WP70PixelMatchL1SeededFilter
    +hltEle26WP70PMS2L1SeededFilter
    +HLTGsfElectronL1SeededSequence
    +hltEle26WP70GsfOneOEMinusOneOPL1SeededFilter
    +hltEle26WP70GsfDetaL1SeededFilter
    +hltEle26WP70GsfDphiL1SeededFilter
    +hltEle26WP70BestGsfNLayerITL1SeededFilter
    +hltEle26WP70BestGsfChi2L1SeededFilter
    +hltEgammaEleL1TrkIsoL1Seeded
    +hltEle26WP70GsfTrackIsoFromL1TracksL1SeededFilter
    +HLTTrackingSequence
    +hltEgammaEleGsfTrackIsoL1Seeded
    +hltEle26WP70GsfTrackIsoL1SeededFilter
)

import FWCore.ParameterSet.Config as cms

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
from ..tasks.HLTEle32WPTightUnseededTask_cfi import *

HLTEle32WPTightUnseededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForSingleEleIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltEG32EtUnseededFilter +
    hltEle32WPTightClusterShapeUnseededFilter +
    hltEle32WPTightClusterShapeSigmavvUnseededFilter +
    hltEle32WPTightClusterShapeSigmawwUnseededFilter +
    hltEle32WPTightHgcalHEUnseededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltEle32WPTightHEUnseededFilter +
    hltEle32WPTightEcalIsoUnseededFilter +
    hltEle32WPTightHgcalIsoUnseededFilter +
    HLTPFHcalClusteringForEgamma +
    hltEle32WPTightHcalIsoUnseededFilter +
    HLTElePixelMatchUnseededSequence +
    hltEle32WPTightPixelMatchUnseededFilter +
    hltEle32WPTightPMS2UnseededFilter +
    HLTGsfElectronUnseededSequence +
    hltEle32WPTightGsfOneOEMinusOneOPUnseededFilter +
    hltEle32WPTightGsfDetaUnseededFilter +
    hltEle32WPTightGsfDphiUnseededFilter +
    hltEle32WPTightBestGsfNLayerITUnseededFilter +
    hltEle32WPTightBestGsfChi2UnseededFilter +
    hltEle32WPTightGsfTrackIsoFromL1TracksUnseededFilter +
    HLTTrackingV61Sequence +
    hltEle32WPTightGsfTrackIsoUnseededFilter,
    HLTEle32WPTightUnseededTask
)

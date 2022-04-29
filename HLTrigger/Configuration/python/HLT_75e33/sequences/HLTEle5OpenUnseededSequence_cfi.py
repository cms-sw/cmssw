import FWCore.ParameterSet.Config as cms

from ..modules.hltEG5EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEle5DphiUnseededFilter_cfi import *
from ..modules.hltEle5WPTightBestGsfChi2UnseededFilter_cfi import *
from ..modules.hltEle5WPTightBestGsfNLayerITUnseededFilter_cfi import *
from ..modules.hltEle5WPTightClusterShapeSigmavvUnseededFilter_cfi import *
from ..modules.hltEle5WPTightClusterShapeSigmawwUnseededFilter_cfi import *
from ..modules.hltEle5WPTightClusterShapeUnseededFilter_cfi import *
from ..modules.hltEle5WPTightEcalIsoUnseededFilter_cfi import *
from ..modules.hltEle5WPTightGsfDetaUnseededFilter_cfi import *
from ..modules.hltEle5WPTightGsfDphiUnseededFilter_cfi import *
from ..modules.hltEle5WPTightGsfOneOEMinusOneOPUnseededFilter_cfi import *
from ..modules.hltEle5WPTightGsfTrackIsoFromL1TracksUnseededFilter_cfi import *
from ..modules.hltEle5WPTightGsfTrackIsoUnseededFilter_cfi import *
from ..modules.hltEle5WPTightHcalIsoUnseededFilter_cfi import *
from ..modules.hltEle5WPTightHEUnseededFilter_cfi import *
from ..modules.hltEle5WPTightHgcalHEUnseededFilter_cfi import *
from ..modules.hltEle5WPTightHgcalIsoUnseededFilter_cfi import *
from ..modules.hltEle5WPTightPixelMatchUnseededFilter_cfi import *
from ..modules.hltEle5WPTightPMS2UnseededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchUnseededSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTGsfElectronUnseededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..tasks.HLTEle5OpenUnseededTask_cfi import *

HLTEle5OpenUnseededSequence = cms.Sequence(
    HLTL1Sequence +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltEG5EtUnseededFilter +
    cms.ignore(hltEle5WPTightClusterShapeUnseededFilter) +
    cms.ignore(hltEle5WPTightClusterShapeSigmavvUnseededFilter) +
    cms.ignore(hltEle5WPTightClusterShapeSigmawwUnseededFilter) +
    cms.ignore(hltEle5WPTightHgcalHEUnseededFilter) +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    cms.ignore(hltEle5WPTightHEUnseededFilter) +
    cms.ignore(hltEle5WPTightEcalIsoUnseededFilter) +
    cms.ignore(hltEle5WPTightHgcalIsoUnseededFilter) +
    HLTPFHcalClusteringForEgamma +
    cms.ignore(hltEle5WPTightHcalIsoUnseededFilter) +
    HLTElePixelMatchUnseededSequence +
    cms.ignore(hltEle5WPTightPixelMatchUnseededFilter) +
    cms.ignore(hltEle5WPTightPMS2UnseededFilter) +
    HLTGsfElectronUnseededSequence +
    cms.ignore(hltEle5WPTightGsfOneOEMinusOneOPUnseededFilter) +
    cms.ignore(hltEle5WPTightGsfDetaUnseededFilter) +
    cms.ignore(hltEle5WPTightGsfDphiUnseededFilter) +
    cms.ignore(hltEle5WPTightBestGsfNLayerITUnseededFilter) +
    cms.ignore(hltEle5WPTightBestGsfChi2UnseededFilter) +
    hltEle5DphiUnseededFilter +
    cms.ignore(hltEle5WPTightGsfTrackIsoFromL1TracksUnseededFilter) +
    HLTTrackingV61Sequence +
    cms.ignore(hltEle5WPTightGsfTrackIsoUnseededFilter),
    HLTEle5OpenUnseededTask
)

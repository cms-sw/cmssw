import FWCore.ParameterSet.Config as cms

from ..modules.hltEG5EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEle5DphiL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightBestGsfChi2L1SeededFilter_cfi import *
from ..modules.hltEle5WPTightBestGsfNLayerITL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightClusterShapeL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightEcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightGsfDetaL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightGsfDphiL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightGsfOneOEMinusOneOPL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightGsfTrackIsoFromL1TracksL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightGsfTrackIsoL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightHcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightHEL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightHgcalHEL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightHgcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightPixelMatchL1SeededFilter_cfi import *
from ..modules.hltEle5WPTightPMS2L1SeededFilter_cfi import *
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
from ..tasks.HLTEle5OpenL1SeededTask_cfi import *

HLTEle5OpenL1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG5EtL1SeededFilter +
    cms.ignore(hltEle5WPTightClusterShapeL1SeededFilter) +
    cms.ignore(hltEle5WPTightClusterShapeSigmavvL1SeededFilter) +
    cms.ignore(hltEle5WPTightClusterShapeSigmawwL1SeededFilter) +
    cms.ignore(hltEle5WPTightHgcalHEL1SeededFilter) +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    cms.ignore(hltEle5WPTightHEL1SeededFilter) +
    cms.ignore(hltEle5WPTightEcalIsoL1SeededFilter) +
    cms.ignore(hltEle5WPTightHgcalIsoL1SeededFilter) +
    HLTPFHcalClusteringForEgamma +
    cms.ignore(hltEle5WPTightHcalIsoL1SeededFilter) +
    HLTElePixelMatchL1SeededSequence +
    cms.ignore(hltEle5WPTightPixelMatchL1SeededFilter) +
    cms.ignore(hltEle5WPTightPMS2L1SeededFilter) +
    HLTGsfElectronL1SeededSequence +
    cms.ignore(hltEle5WPTightGsfOneOEMinusOneOPL1SeededFilter) +
    cms.ignore(hltEle5WPTightGsfDetaL1SeededFilter) +
    cms.ignore(hltEle5WPTightGsfDphiL1SeededFilter) +
    cms.ignore(hltEle5WPTightBestGsfNLayerITL1SeededFilter) +
    cms.ignore(hltEle5WPTightBestGsfChi2L1SeededFilter) +
    hltEle5DphiL1SeededFilter +
    cms.ignore(hltEle5WPTightGsfTrackIsoFromL1TracksL1SeededFilter) +
    HLTTrackingV61Sequence +
    cms.ignore(hltEle5WPTightGsfTrackIsoL1SeededFilter),
    HLTEle5OpenL1SeededTask
)

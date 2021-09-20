import FWCore.ParameterSet.Config as cms

from ..modules.hltEG32EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForSingleEleIsolatedFilter_cfi import *
from ..modules.hltEle32WPTightBestGsfChi2L1SeededFilter_cfi import *
from ..modules.hltEle32WPTightBestGsfNLayerITL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightEcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfDetaL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfDphiL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightGsfTrackIsoL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightHcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightHEL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightHgcalHEL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightHgcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightPixelMatchL1SeededFilter_cfi import *
from ..modules.hltEle32WPTightPMS2L1SeededFilter_cfi import *
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
from ..tasks.HLTEle32WPTightL1SeededTask_cfi import *

HLTEle32WPTightL1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForSingleEleIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG32EtL1SeededFilter +
    hltEle32WPTightClusterShapeL1SeededFilter +
    hltEle32WPTightClusterShapeSigmavvL1SeededFilter +
    hltEle32WPTightClusterShapeSigmawwL1SeededFilter +
    hltEle32WPTightHgcalHEL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltEle32WPTightHEL1SeededFilter +
    hltEle32WPTightEcalIsoL1SeededFilter +
    hltEle32WPTightHgcalIsoL1SeededFilter +
    HLTPFHcalClusteringForEgamma +
    hltEle32WPTightHcalIsoL1SeededFilter +
    HLTElePixelMatchL1SeededSequence +
    hltEle32WPTightPixelMatchL1SeededFilter +
    hltEle32WPTightPMS2L1SeededFilter +
    HLTGsfElectronL1SeededSequence +
    hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter +
    hltEle32WPTightGsfDetaL1SeededFilter +
    hltEle32WPTightGsfDphiL1SeededFilter +
    hltEle32WPTightBestGsfNLayerITL1SeededFilter +
    hltEle32WPTightBestGsfChi2L1SeededFilter +
    hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter +
    HLTTrackingV61Sequence +
    hltEle32WPTightGsfTrackIsoL1SeededFilter,
    HLTEle32WPTightL1SeededTask
)

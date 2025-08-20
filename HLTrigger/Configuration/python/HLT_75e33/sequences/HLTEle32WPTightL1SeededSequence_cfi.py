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

from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
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
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

HLTEle32WPTightL1SeededSequence = cms.Sequence(
    hltEGL1SeedsForSingleEleIsolatedFilter
    +HLTDoFullUnpackingEgammaEcalL1SeededSequence
    +HLTPFClusteringForEgammaL1SeededSequence
    +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
    +hltEgammaCandidatesL1Seeded
    +hltEgammaCandidatesWrapperL1Seeded
    +hltEG32EtL1SeededFilter
    +hltEgammaClusterShapeL1Seeded
    +hltEle32WPTightClusterShapeL1SeededFilter
    +hltEgammaHGCALIDVarsL1Seeded
    +hltEle32WPTightClusterShapeSigmavvL1SeededFilter
    +hltEle32WPTightClusterShapeSigmawwL1SeededFilter
    +hltEle32WPTightHgcalHEL1SeededFilter
    +HLTEGammaDoLocalHcalSequence
    +HLTFastJetForEgammaSequence
    +hltEgammaHoverEL1Seeded
    +hltEle32WPTightHEL1SeededFilter
    +hltEgammaEcalPFClusterIsoL1Seeded
    +hltEle32WPTightEcalIsoL1SeededFilter
    +hltEgammaHGCalLayerClusterIsoL1Seeded
    +hltEle32WPTightHgcalIsoL1SeededFilter
    +HLTPFHcalClusteringForEgammaSequence
    +hltEgammaHcalPFClusterIsoL1Seeded
    +hltEle32WPTightHcalIsoL1SeededFilter
    +HLTElePixelMatchL1SeededSequence
    +hltEle32WPTightPixelMatchL1SeededFilter
    +hltEle32WPTightPMS2L1SeededFilter
    +HLTGsfElectronL1SeededSequence
    +hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter
    +hltEle32WPTightGsfDetaL1SeededFilter
    +hltEle32WPTightGsfDphiL1SeededFilter
    +hltEle32WPTightBestGsfNLayerITL1SeededFilter
    +hltEle32WPTightBestGsfChi2L1SeededFilter
    +hltEgammaEleL1TrkIsoL1Seeded
    +hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter
    +HLTTrackingSequence
    +hltEgammaEleGsfTrackIsoL1Seeded
    +hltEle32WPTightGsfTrackIsoL1SeededFilter

)


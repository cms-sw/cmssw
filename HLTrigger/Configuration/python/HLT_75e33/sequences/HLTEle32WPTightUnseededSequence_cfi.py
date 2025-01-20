import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchUnseededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTGsfElectronUnseededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoUnseeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *
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

HLTEle32WPTightUnseededSequence = cms.Sequence(HLTL1Sequence
    +hltEGL1SeedsForSingleEleIsolatedFilter
    +HLTDoFullUnpackingEgammaEcalSequence
    +HLTPFClusteringForEgammaUnseededSequence
    +HLTHgcalTiclPFClusteringForEgammaUnseededSequence
    +hltEgammaCandidatesUnseeded
    +hltEgammaCandidatesWrapperUnseeded                                           
    +hltEG32EtUnseededFilter
    +hltEgammaClusterShapeUnseeded
    +hltEle32WPTightClusterShapeUnseededFilter
    +hltEgammaHGCALIDVarsUnseeded
    +hltEle32WPTightClusterShapeSigmavvUnseededFilter
    +hltEle32WPTightClusterShapeSigmawwUnseededFilter
    +hltEle32WPTightHgcalHEUnseededFilter
    +HLTEGammaDoLocalHcalSequence
    +HLTFastJetForEgammaSequence
    +hltEgammaHoverEUnseeded                                            
    +hltEle32WPTightHEUnseededFilter
    +hltEgammaEcalPFClusterIsoUnseeded
    +hltEle32WPTightEcalIsoUnseededFilter
    +hltEgammaHGCalLayerClusterIsoUnseeded
    +hltEle32WPTightHgcalIsoUnseededFilter                                           
    +HLTPFHcalClusteringForEgammaSequence
    +hltEgammaHcalPFClusterIsoUnseeded
    +hltEle32WPTightHcalIsoUnseededFilter                                           
    +HLTElePixelMatchUnseededSequence
    +hltEle32WPTightPixelMatchUnseededFilter
    +hltEle32WPTightPMS2UnseededFilter
    +HLTGsfElectronUnseededSequence
    +hltEle32WPTightGsfOneOEMinusOneOPUnseededFilter
    +hltEle32WPTightGsfDetaUnseededFilter
    +hltEle32WPTightGsfDphiUnseededFilter
    +hltEle32WPTightBestGsfNLayerITUnseededFilter
    +hltEle32WPTightBestGsfChi2UnseededFilter
    +hltEgammaEleL1TrkIsoUnseeded
    +hltEle32WPTightGsfTrackIsoFromL1TracksUnseededFilter
    +HLTTrackingSequence
    +hltEgammaEleGsfTrackIsoUnseeded
    +hltEle32WPTightGsfTrackIsoUnseededFilter
)

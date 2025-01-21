import FWCore.ParameterSet.Config as cms

from ..modules.hltEG5EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaR9Unseeded_cfi import *
from ..modules.hltEgammaHollowTrackIsoUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoUnseeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *
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

HLTEle5OpenUnseededSequence = cms.Sequence(HLTL1Sequence
                                           +HLTDoFullUnpackingEgammaEcalSequence
                                           +HLTEGammaDoLocalHcalSequence
                                           +HLTPFClusteringForEgammaUnseededSequence
                                           +HLTHgcalTiclPFClusteringForEgammaUnseededSequence
                                           +HLTFastJetForEgammaSequence
                                           +HLTPFHcalClusteringForEgammaSequence
                                           +HLTElePixelMatchUnseededSequence
                                           +HLTTrackingSequence
                                           +HLTGsfElectronUnseededSequence
                                           +hltEgammaCandidatesUnseeded
                                           +hltEgammaClusterShapeUnseeded
                                           +hltEgammaR9Unseeded
                                           +hltEgammaHGCALIDVarsUnseeded
                                           +hltEgammaHoverEUnseeded
                                           +hltEgammaEcalPFClusterIsoUnseeded
                                           +hltEgammaHGCalLayerClusterIsoUnseeded
                                           +hltEgammaHcalPFClusterIsoUnseeded
                                           +hltEgammaEleGsfTrackIsoUnseeded
                                           +hltEgammaEleL1TrkIsoUnseeded
                                           +hltEgammaHollowTrackIsoUnseeded
                                           +hltEgammaCandidatesWrapperUnseeded
                                           +hltEG5EtUnseededFilter
                                           +hltEle5DphiUnseededFilter
                                           +cms.ignore(hltEle5WPTightClusterShapeUnseededFilter)
                                           +cms.ignore(hltEle5WPTightClusterShapeSigmavvUnseededFilter)
                                           +cms.ignore(hltEle5WPTightClusterShapeSigmawwUnseededFilter)
                                           +cms.ignore(hltEle5WPTightHgcalHEUnseededFilter)
                                           +cms.ignore(hltEle5WPTightHEUnseededFilter)
                                           +cms.ignore(hltEle5WPTightEcalIsoUnseededFilter)
                                           +cms.ignore(hltEle5WPTightHgcalIsoUnseededFilter)
                                           +cms.ignore(hltEle5WPTightHcalIsoUnseededFilter)
                                           +cms.ignore(hltEle5WPTightPixelMatchUnseededFilter)
                                           +cms.ignore(hltEle5WPTightPMS2UnseededFilter)
                                           +cms.ignore(hltEle5WPTightGsfOneOEMinusOneOPUnseededFilter)
                                           +cms.ignore(hltEle5WPTightGsfDetaUnseededFilter)
                                           +cms.ignore(hltEle5WPTightGsfDphiUnseededFilter)
                                           +cms.ignore(hltEle5WPTightBestGsfNLayerITUnseededFilter)
                                           +cms.ignore(hltEle5WPTightBestGsfChi2UnseededFilter)
                                           +cms.ignore(hltEle5WPTightGsfTrackIsoFromL1TracksUnseededFilter)
                                           +cms.ignore(hltEle5WPTightGsfTrackIsoUnseededFilter))

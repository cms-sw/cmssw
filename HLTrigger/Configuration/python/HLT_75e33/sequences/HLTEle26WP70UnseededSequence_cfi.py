import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoUnseeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *
from ..modules.hltEG26EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEGL1SeedsForSingleEleIsolatedFilter_cfi import *
from ..modules.hltEle26WP70BestGsfChi2UnseededFilter_cfi import *
from ..modules.hltEle26WP70BestGsfNLayerITUnseededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeSigmavvUnseededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeSigmawwUnseededFilter_cfi import *
from ..modules.hltEle26WP70ClusterShapeUnseededFilter_cfi import *
from ..modules.hltEle26WP70EcalIsoUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfDetaUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfDphiUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfOneOEMinusOneOPUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfTrackIsoFromL1TracksUnseededFilter_cfi import *
from ..modules.hltEle26WP70GsfTrackIsoUnseededFilter_cfi import *
from ..modules.hltEle26WP70HcalIsoUnseededFilter_cfi import *
from ..modules.hltEle26WP70HEUnseededFilter_cfi import *
from ..modules.hltEle26WP70HgcalHEUnseededFilter_cfi import *
from ..modules.hltEle26WP70HgcalIsoUnseededFilter_cfi import *
from ..modules.hltEle26WP70PixelMatchUnseededFilter_cfi import *
from ..modules.hltEle26WP70PMS2UnseededFilter_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchUnseededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTGsfElectronUnseededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *

HLTEle26WP70UnseededSequence = cms.Sequence(hltEGL1SeedsForSingleEleIsolatedFilter
                                            +HLTDoFullUnpackingEgammaEcalSequence
                                            +HLTPFClusteringForEgammaUnseededSequence
                                            +HLTHgcalTiclPFClusteringForEgammaUnseededSequence
                                            +hltEgammaCandidatesUnseeded
                                            +hltEgammaCandidatesWrapperUnseeded
                                            +hltEG26EtUnseededFilter
                                            +hltEgammaClusterShapeUnseeded
                                            +hltEle26WP70ClusterShapeUnseededFilter
                                            +hltEgammaHGCALIDVarsUnseeded
                                            +hltEle26WP70ClusterShapeSigmavvUnseededFilter
                                            +hltEle26WP70ClusterShapeSigmawwUnseededFilter
                                            +hltEle26WP70HgcalHEUnseededFilter
                                            +HLTEGammaDoLocalHcalSequence    
                                            +HLTFastJetForEgammaSequence
                                            +hltEgammaHoverEUnseeded
                                            +hltEle26WP70HEUnseededFilter
                                            +hltEgammaEcalPFClusterIsoUnseeded
                                            +hltEle26WP70EcalIsoUnseededFilter
                                            +hltEgammaHGCalLayerClusterIsoUnseeded
                                            +hltEle26WP70HgcalIsoUnseededFilter
                                            +HLTPFHcalClusteringForEgammaSequence
                                            +hltEgammaHcalPFClusterIsoUnseeded
                                            +hltEle26WP70HcalIsoUnseededFilter
                                            +HLTElePixelMatchUnseededSequence
                                            +hltEle26WP70PixelMatchUnseededFilter
                                            +hltEle26WP70PMS2UnseededFilter
                                            +HLTGsfElectronUnseededSequence
                                            +hltEle26WP70GsfOneOEMinusOneOPUnseededFilter
                                            +hltEle26WP70GsfDetaUnseededFilter
                                            +hltEle26WP70GsfDphiUnseededFilter
                                            +hltEle26WP70BestGsfNLayerITUnseededFilter
                                            +hltEle26WP70BestGsfChi2UnseededFilter
                                            +hltEgammaEleL1TrkIsoUnseeded
                                            +hltEle26WP70GsfTrackIsoFromL1TracksUnseededFilter
                                            +HLTTrackingSequence
                                            +hltEgammaEleGsfTrackIsoUnseeded
                                            +hltEle26WP70GsfTrackIsoUnseededFilter)

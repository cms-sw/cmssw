import FWCore.ParameterSet.Config as cms

from ..modules.hltDiEG23EtUnseededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdClusterShapeSigmavvUnseededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdClusterShapeSigmawwUnseededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdClusterShapeUnseededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdEcalIsoUnseededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdHcalIsoUnseededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdHEUnseededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdHgcalHEUnseededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdHgcalIsoUnseededFilter_cfi import *
from ..modules.hltEG30EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEGL1SeedsForDoublePhotonIsolatedFilter_cfi import *
from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *


HLTDiphoton3023IsoCaloIdUnseededSequence = cms.Sequence(hltEGL1SeedsForDoublePhotonIsolatedFilter
                                                        +HLTDoFullUnpackingEgammaEcalSequence
                                                        +HLTPFClusteringForEgammaUnseededSequence
                                                        +HLTHgcalTiclPFClusteringForEgammaUnseededSequence
                                                        +hltEgammaCandidatesUnseeded
                                                        +hltEgammaCandidatesWrapperUnseeded
                                                        +hltEG30EtUnseededFilter
                                                        +hltDiEG23EtUnseededFilter
                                                        +hltEgammaClusterShapeUnseeded
                                                        +hltDiEG3023IsoCaloIdClusterShapeUnseededFilter
                                                        +hltEgammaHGCALIDVarsUnseeded
                                                        +hltDiEG3023IsoCaloIdClusterShapeSigmavvUnseededFilter
                                                        +hltDiEG3023IsoCaloIdClusterShapeSigmawwUnseededFilter
                                                        +hltDiEG3023IsoCaloIdHgcalHEUnseededFilter
                                                        +HLTEGammaDoLocalHcalSequence
                                                        +HLTFastJetForEgammaSequence
                                                        +hltEgammaHoverEUnseeded
                                                        +hltDiEG3023IsoCaloIdHEUnseededFilter
                                                        +hltEgammaEcalPFClusterIsoUnseeded
                                                        +hltDiEG3023IsoCaloIdEcalIsoUnseededFilter
                                                        +hltEgammaHGCalLayerClusterIsoUnseeded
                                                        +hltDiEG3023IsoCaloIdHgcalIsoUnseededFilter
                                                        +HLTPFHcalClusteringForEgammaSequence
                                                        +hltEgammaHcalPFClusterIsoUnseeded
                                                        +hltDiEG3023IsoCaloIdHcalIsoUnseededFilter)

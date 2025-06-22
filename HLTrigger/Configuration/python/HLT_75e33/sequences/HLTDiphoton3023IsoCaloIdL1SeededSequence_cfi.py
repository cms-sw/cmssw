import FWCore.ParameterSet.Config as cms

from ..modules.hltDiEG23EtL1SeededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdClusterShapeL1SeededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdEcalIsoL1SeededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdHcalIsoL1SeededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdHEL1SeededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdHgcalHEL1SeededFilter_cfi import *
from ..modules.hltDiEG3023IsoCaloIdHgcalIsoL1SeededFilter_cfi import *
from ..modules.hltEG30EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForDoublePhotonIsolatedFilter_cfi import *
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *

HLTDiphoton3023IsoCaloIdL1SeededSequence = cms.Sequence(hltEGL1SeedsForDoublePhotonIsolatedFilter
                                                        +HLTDoFullUnpackingEgammaEcalL1SeededSequence
                                                        +HLTPFClusteringForEgammaL1SeededSequence
                                                        +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
                                                        +hltEgammaCandidatesL1Seeded
                                                        +hltEgammaCandidatesWrapperL1Seeded
                                                        +hltEG30EtL1SeededFilter
                                                        +hltDiEG23EtL1SeededFilter
                                                        +hltEgammaClusterShapeL1Seeded
                                                        +hltDiEG3023IsoCaloIdClusterShapeL1SeededFilter
                                                        +hltEgammaHGCALIDVarsL1Seeded
                                                        +hltDiEG3023IsoCaloIdClusterShapeSigmavvL1SeededFilter
                                                        +hltDiEG3023IsoCaloIdClusterShapeSigmawwL1SeededFilter
                                                        +hltDiEG3023IsoCaloIdHgcalHEL1SeededFilter
                                                        +HLTEGammaDoLocalHcalSequence
                                                        +HLTFastJetForEgammaSequence
                                                        +hltEgammaHoverEL1Seeded
                                                        +hltDiEG3023IsoCaloIdHEL1SeededFilter
                                                        +hltEgammaEcalPFClusterIsoL1Seeded
                                                        +hltDiEG3023IsoCaloIdEcalIsoL1SeededFilter
                                                        +hltEgammaHGCalLayerClusterIsoL1Seeded
                                                        +hltDiEG3023IsoCaloIdHgcalIsoL1SeededFilter
                                                        +HLTPFHcalClusteringForEgammaSequence
                                                        +hltEgammaHcalPFClusterIsoL1Seeded
                                                        +hltDiEG3023IsoCaloIdHcalIsoL1SeededFilter)

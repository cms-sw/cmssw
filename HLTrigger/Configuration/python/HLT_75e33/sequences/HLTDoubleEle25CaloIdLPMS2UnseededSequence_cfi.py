import FWCore.ParameterSet.Config as cms

from ..modules.hltDiEG25CaloIdLClusterShapeSigmavvUnseededFilter_cfi import *
from ..modules.hltDiEG25CaloIdLClusterShapeUnseededFilter_cfi import *
from ..modules.hltDiEG25CaloIdLHEUnseededFilter_cfi import *
from ..modules.hltDiEG25CaloIdLHgcalHEUnseededFilter_cfi import *
from ..modules.hltDiEG25EtUnseededFilter_cfi import *
from ..modules.hltDiEle25CaloIdLPixelMatchUnseededFilter_cfi import *
from ..modules.hltDiEle25CaloIdLPMS2UnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEGL1SeedsForDoubleEleNonIsolatedFilter_cfi import *
from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchUnseededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *

HLTDoubleEle25CaloIdLPMS2UnseededSequence = cms.Sequence(HLTL1Sequence
                                                         +hltEGL1SeedsForDoubleEleNonIsolatedFilter
                                                         +HLTDoFullUnpackingEgammaEcalSequence
                                                         +HLTPFClusteringForEgammaUnseededSequence
                                                         +HLTHgcalTiclPFClusteringForEgammaUnseededSequence
                                                         +hltEgammaCandidatesUnseeded
                                                         +hltEgammaCandidatesWrapperUnseeded
                                                         +hltDiEG25EtUnseededFilter
                                                         +hltEgammaClusterShapeUnseeded
                                                         +hltDiEG25CaloIdLClusterShapeUnseededFilter
                                                         +hltEgammaHGCALIDVarsUnseeded
                                                         +hltDiEG25CaloIdLClusterShapeSigmavvUnseededFilter
                                                         +hltDiEG25CaloIdLHgcalHEUnseededFilter
                                                         +HLTEGammaDoLocalHcalSequence
                                                         +HLTFastJetForEgammaSequence
                                                         +hltEgammaHoverEUnseeded
                                                         +hltDiEG25CaloIdLHEUnseededFilter
                                                         +HLTElePixelMatchUnseededSequence
                                                         +hltDiEle25CaloIdLPixelMatchUnseededFilter
                                                         +hltDiEle25CaloIdLPMS2UnseededFilter)

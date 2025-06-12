import FWCore.ParameterSet.Config as cms

from ..modules.hltDiEG25CaloIdLClusterShapeL1SeededFilter_cfi import *
from ..modules.hltDiEG25CaloIdLClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltDiEG25CaloIdLHEL1SeededFilter_cfi import *
from ..modules.hltDiEG25CaloIdLHgcalHEL1SeededFilter_cfi import *
from ..modules.hltDiEG25EtL1SeededFilter_cfi import *
from ..modules.hltDiEle25CaloIdLPixelMatchL1SeededFilter_cfi import *
from ..modules.hltDiEle25CaloIdLPMS2L1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForDoubleEleNonIsolatedFilter_cfi import *
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *

HLTDoubleEle25CaloIdLPMS2L1SeededSequence = cms.Sequence(hltEGL1SeedsForDoubleEleNonIsolatedFilter
                                                         +HLTDoFullUnpackingEgammaEcalL1SeededSequence
                                                         +HLTPFClusteringForEgammaL1SeededSequence
                                                         +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
                                                         +hltEgammaCandidatesL1Seeded
                                                         +hltEgammaCandidatesWrapperL1Seeded
                                                         +hltDiEG25EtL1SeededFilter
                                                         +hltEgammaClusterShapeL1Seeded
                                                         +hltDiEG25CaloIdLClusterShapeL1SeededFilter
                                                         +hltEgammaHGCALIDVarsL1Seeded
                                                         +hltDiEG25CaloIdLClusterShapeSigmavvL1SeededFilter
                                                         +hltDiEG25CaloIdLHgcalHEL1SeededFilter
                                                         +HLTEGammaDoLocalHcalSequence
                                                         +hltEgammaHoverEL1Seeded
                                                         +hltDiEG25CaloIdLHEL1SeededFilter
                                                         +HLTElePixelMatchL1SeededSequence
                                                         +hltDiEle25CaloIdLPixelMatchL1SeededFilter
                                                         +hltDiEle25CaloIdLPMS2L1SeededFilter)

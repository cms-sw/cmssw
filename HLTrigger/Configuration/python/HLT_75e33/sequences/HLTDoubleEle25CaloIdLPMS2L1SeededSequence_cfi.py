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
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..tasks.HLTDoubleEle25CaloIdLPMS2L1SeededTask_cfi import *

HLTDoubleEle25CaloIdLPMS2L1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForDoubleEleNonIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltDiEG25EtL1SeededFilter +
    hltDiEG25CaloIdLClusterShapeL1SeededFilter +
    hltDiEG25CaloIdLClusterShapeSigmavvL1SeededFilter +
    hltDiEG25CaloIdLHgcalHEL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltDiEG25CaloIdLHEL1SeededFilter +
    HLTElePixelMatchL1SeededSequence +
    hltDiEle25CaloIdLPixelMatchL1SeededFilter +
    hltDiEle25CaloIdLPMS2L1SeededFilter,
    HLTDoubleEle25CaloIdLPMS2L1SeededTask
)

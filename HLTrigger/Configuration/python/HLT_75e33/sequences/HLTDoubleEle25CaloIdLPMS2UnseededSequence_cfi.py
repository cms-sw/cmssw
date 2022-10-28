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
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchUnseededSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseeded_cfi import *
from ..tasks.HLTDoubleEle25CaloIdLPMS2UnseededTask_cfi import *

HLTDoubleEle25CaloIdLPMS2UnseededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForDoubleEleNonIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltDiEG25EtUnseededFilter +
    hltDiEG25CaloIdLClusterShapeUnseededFilter +
    hltDiEG25CaloIdLClusterShapeSigmavvUnseededFilter +
    hltDiEG25CaloIdLHgcalHEUnseededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltDiEG25CaloIdLHEUnseededFilter +
    HLTElePixelMatchUnseededSequence +
    hltDiEle25CaloIdLPixelMatchUnseededFilter +
    hltDiEle25CaloIdLPMS2UnseededFilter,
    HLTDoubleEle25CaloIdLPMS2UnseededTask
)

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
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..tasks.HLTDiphoton3023IsoCaloIdL1SeededTask_cfi import *

HLTDiphoton3023IsoCaloIdL1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForDoublePhotonIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG30EtL1SeededFilter +
    hltDiEG23EtL1SeededFilter +
    hltDiEG3023IsoCaloIdClusterShapeL1SeededFilter +
    hltDiEG3023IsoCaloIdClusterShapeSigmavvL1SeededFilter +
    hltDiEG3023IsoCaloIdClusterShapeSigmawwL1SeededFilter +
    hltDiEG3023IsoCaloIdHgcalHEL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltDiEG3023IsoCaloIdHEL1SeededFilter +
    hltDiEG3023IsoCaloIdEcalIsoL1SeededFilter +
    hltDiEG3023IsoCaloIdHgcalIsoL1SeededFilter +
    HLTPFHcalClusteringForEgamma +
    hltDiEG3023IsoCaloIdHcalIsoL1SeededFilter,
    HLTDiphoton3023IsoCaloIdL1SeededTask
)

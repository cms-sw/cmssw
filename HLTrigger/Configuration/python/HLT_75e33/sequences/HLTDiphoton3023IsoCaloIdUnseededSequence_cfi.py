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
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..tasks.HLTDiphoton3023IsoCaloIdUnseededTask_cfi import *

HLTDiphoton3023IsoCaloIdUnseededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForDoublePhotonIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltEG30EtUnseededFilter +
    hltDiEG23EtUnseededFilter +
    hltDiEG3023IsoCaloIdClusterShapeUnseededFilter +
    hltDiEG3023IsoCaloIdClusterShapeSigmavvUnseededFilter +
    hltDiEG3023IsoCaloIdClusterShapeSigmawwUnseededFilter +
    hltDiEG3023IsoCaloIdHgcalHEUnseededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltDiEG3023IsoCaloIdHEUnseededFilter +
    hltDiEG3023IsoCaloIdEcalIsoUnseededFilter +
    hltDiEG3023IsoCaloIdHgcalIsoUnseededFilter +
    HLTPFHcalClusteringForEgamma +
    hltDiEG3023IsoCaloIdHcalIsoUnseededFilter,
    HLTDiphoton3023IsoCaloIdUnseededTask
)

import FWCore.ParameterSet.Config as cms

from ..modules.hltEG187EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltPhoton187HEUnseededFilter_cfi import *
from ..modules.hltPhoton187HgcalHEUnseededFilter_cfi import *
from ..modules.L1TkEmSingle51Filter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseeded_cfi import *
from ..tasks.HLTPhoton187UnseededTask_cfi import *

HLTPhoton187UnseededSequence = cms.Sequence(
    HLTL1Sequence +
    L1TkEmSingle51Filter +
    HLTDoFullUnpackingEgammaEcalSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltEG187EtUnseededFilter +
    hltPhoton187HgcalHEUnseededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltPhoton187HEUnseededFilter,
    HLTPhoton187UnseededTask
)

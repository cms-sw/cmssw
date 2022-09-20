import FWCore.ParameterSet.Config as cms

from ..modules.hltEG187EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltPhoton187HEL1SeededFilter_cfi import *
from ..modules.hltPhoton187HgcalHEL1SeededFilter_cfi import *
from ..modules.L1TkEmSingle51Filter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..tasks.HLTPhoton187L1SeededTask_cfi import *

HLTPhoton187L1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    l1tTkEmSingle51Filter +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG187EtL1SeededFilter +
    hltPhoton187HgcalHEL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltPhoton187HEL1SeededFilter,
    HLTPhoton187L1SeededTask
)

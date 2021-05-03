import FWCore.ParameterSet.Config as cms

from ..modules.hltEG100EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltPhoton100HEL1SeededFilter_cfi import *
from ..modules.hltPhoton100HgcalHEL1SeededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..tasks.HLTPhoton100OpenL1SeededTask_cfi import *

HLTPhoton100OpenL1SeededSequence = cms.Sequence(HLTL1Sequence+HLTDoFullUnpackingEgammaEcalL1SeededSequence+HLTPFClusteringForEgammaL1Seeded+HLTHgcalTiclPFClusteringForEgammaL1Seeded+hltEgammaCandidatesWrapperL1Seeded+hltEG100EtL1SeededFilter+hltPhoton100HgcalHEL1SeededFilter+HLTDoLocalHcalSequence+HLTFastJetForEgamma+hltPhoton100HEL1SeededFilter, HLTPhoton100OpenL1SeededTask)

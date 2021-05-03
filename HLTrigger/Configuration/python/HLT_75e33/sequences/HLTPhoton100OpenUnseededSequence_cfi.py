import FWCore.ParameterSet.Config as cms

from ..modules.hltEG100EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltPhoton100HEUnseededFilter_cfi import *
from ..modules.hltPhoton100HgcalHEUnseededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseeded_cfi import *
from ..tasks.HLTPhoton100OpenUnseededTask_cfi import *

HLTPhoton100OpenUnseededSequence = cms.Sequence(HLTL1Sequence+HLTDoFullUnpackingEgammaEcalL1SeededSequence+HLTPFClusteringForEgammaUnseeded+HLTHgcalTiclPFClusteringForEgammaUnseeded+hltEgammaCandidatesWrapperUnseeded+hltEG100EtUnseededFilter+cms.ignore(hltPhoton100HgcalHEUnseededFilter)+HLTDoLocalHcalSequence+HLTFastJetForEgamma+cms.ignore(hltPhoton100HEUnseededFilter), HLTPhoton100OpenUnseededTask)

import FWCore.ParameterSet.Config as cms

from ..modules.hltEG187EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltPhoton187HEUnseededFilter_cfi import *
from ..modules.hltPhoton187HgcalHEUnseededFilter_cfi import *
from ..modules.l1tTkEmSingle51Filter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPhoton187UnseededInnerSequence_cfi import *

HLTPhoton187UnseededSequence = cms.Sequence(HLTL1Sequence+l1tTkEmSingle51Filter+HLTDoFullUnpackingEgammaEcalSequence+HLTEGammaDoLocalHcalSequence+HLTPFClusteringForEgammaUnseededSequence+HLTHgcalTiclPFClusteringForEgammaUnseededSequence+HLTPhoton187UnseededInnerSequence+hltEgammaCandidatesWrapperUnseeded+hltEG187EtUnseededFilter+hltPhoton187HgcalHEUnseededFilter+hltPhoton187HEUnseededFilter)

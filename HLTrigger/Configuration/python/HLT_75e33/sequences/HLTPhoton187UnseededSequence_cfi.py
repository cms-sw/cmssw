import FWCore.ParameterSet.Config as cms

from ..modules.hltEG187EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltPhoton187HEUnseededFilter_cfi import *
from ..modules.hltPhoton187HgcalHEUnseededFilter_cfi import *
from ..modules.l1tTkEmSingle51Filter_cfi import *
from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *

HLTPhoton187UnseededSequence = cms.Sequence(l1tTkEmSingle51Filter
                                            +HLTDoFullUnpackingEgammaEcalSequence
                                            +HLTPFClusteringForEgammaUnseededSequence
                                            +HLTHgcalTiclPFClusteringForEgammaUnseededSequence
                                            +hltEgammaCandidatesUnseeded
                                            +hltEgammaCandidatesWrapperUnseeded
                                            +hltEG187EtUnseededFilter
                                            +hltEgammaHGCALIDVarsUnseeded
                                            +hltPhoton187HgcalHEUnseededFilter
                                            +HLTEGammaDoLocalHcalSequence
                                            +HLTFastJetForEgammaSequence
                                            +hltEgammaHoverEUnseeded
                                            +hltPhoton187HEUnseededFilter)

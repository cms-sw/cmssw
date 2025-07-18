import FWCore.ParameterSet.Config as cms

from ..modules.hltEG187EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltPhoton187HEL1SeededFilter_cfi import *
from ..modules.hltPhoton187HgcalHEL1SeededFilter_cfi import *
from ..modules.l1tTkEmSingle51Filter_cfi import *
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *

HLTPhoton187L1SeededSequence = cms.Sequence(l1tTkEmSingle51Filter
                                            +HLTDoFullUnpackingEgammaEcalL1SeededSequence
                                            +HLTPFClusteringForEgammaL1SeededSequence
                                            +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
                                            +hltEgammaCandidatesL1Seeded
                                            +hltEgammaCandidatesWrapperL1Seeded
                                            +hltEG187EtL1SeededFilter
                                            +hltEgammaHGCALIDVarsL1Seeded
                                            +hltPhoton187HgcalHEL1SeededFilter
                                            +HLTEGammaDoLocalHcalSequence
                                            +hltEgammaHoverEL1Seeded
                                            +HLTFastJetForEgammaSequence
                                            +hltPhoton187HEL1SeededFilter)

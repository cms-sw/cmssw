import FWCore.ParameterSet.Config as cms

from ..modules.hltEG108EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEGL1SeedsForSinglePhotonIsolatedFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoClusterShapeUnseededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoEcalIsoUnseededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoHcalIsoUnseededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoHEUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *

HLTPhoton108EBTightIDTightIsoUnseededSequence = cms.Sequence(hltEGL1SeedsForSinglePhotonIsolatedFilter
                                                             +HLTDoFullUnpackingEgammaEcalSequence
                                                             +HLTPFClusteringForEgammaUnseededSequence
                                                             +HLTHgcalTiclPFClusteringForEgammaUnseededSequence
                                                             +hltEgammaCandidatesUnseeded
                                                             +hltEgammaCandidatesWrapperUnseeded
                                                             +hltEG108EtUnseededFilter
                                                             +hltEgammaClusterShapeUnseeded
                                                             +hltPhoton108EBTightIDTightIsoClusterShapeUnseededFilter
                                                             +HLTEGammaDoLocalHcalSequence 
                                                             +HLTFastJetForEgammaSequence
                                                             +hltEgammaHoverEUnseeded
                                                             +hltPhoton108EBTightIDTightIsoHEUnseededFilter
                                                             +hltEgammaEcalPFClusterIsoUnseeded
                                                             +hltPhoton108EBTightIDTightIsoEcalIsoUnseededFilter
                                                             +HLTPFHcalClusteringForEgammaSequence
                                                             +hltEgammaHcalPFClusterIsoUnseeded
                                                             +hltPhoton108EBTightIDTightIsoHcalIsoUnseededFilter)

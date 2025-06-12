import FWCore.ParameterSet.Config as cms

from ..modules.hltEG108EtL1SeededFilter_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForSinglePhotonIsolatedFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoClusterShapeL1SeededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoEcalIsoL1SeededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoHcalIsoL1SeededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoHEL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *

HLTPhoton108EBTightIDTightIsoL1SeededSequence = cms.Sequence(
        hltEGL1SeedsForSinglePhotonIsolatedFilter
        +HLTDoFullUnpackingEgammaEcalL1SeededSequence
        +HLTPFClusteringForEgammaL1SeededSequence
        +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
        +hltEgammaCandidatesL1Seeded
        +hltEgammaCandidatesWrapperL1Seeded
        +hltEG108EtL1SeededFilter
        +hltEgammaClusterShapeL1Seeded
        +hltPhoton108EBTightIDTightIsoClusterShapeL1SeededFilter
        +HLTEGammaDoLocalHcalSequence
        +HLTFastJetForEgammaSequence
        +hltEgammaHoverEL1Seeded
        +hltPhoton108EBTightIDTightIsoHEL1SeededFilter
        +hltEgammaEcalPFClusterIsoL1Seeded
        +hltPhoton108EBTightIDTightIsoEcalIsoL1SeededFilter
        +HLTPFHcalClusteringForEgammaSequence
        +hltEgammaHcalPFClusterIsoL1Seeded
        +hltPhoton108EBTightIDTightIsoHcalIsoL1SeededFilter)

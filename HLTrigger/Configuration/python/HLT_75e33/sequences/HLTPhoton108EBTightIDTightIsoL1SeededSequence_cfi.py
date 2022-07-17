import FWCore.ParameterSet.Config as cms

from ..modules.hltEG108EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForSinglePhotonIsolatedFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoClusterShapeL1SeededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoEcalIsoL1SeededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoHcalIsoL1SeededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoHEL1SeededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..tasks.HLTPhoton108EBTightIDTightIsoL1SeededTask_cfi import *

HLTPhoton108EBTightIDTightIsoL1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForSinglePhotonIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG108EtL1SeededFilter +
    hltPhoton108EBTightIDTightIsoClusterShapeL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltPhoton108EBTightIDTightIsoHEL1SeededFilter +
    hltPhoton108EBTightIDTightIsoEcalIsoL1SeededFilter +
    HLTPFHcalClusteringForEgamma +
    hltPhoton108EBTightIDTightIsoHcalIsoL1SeededFilter,
    HLTPhoton108EBTightIDTightIsoL1SeededTask
)

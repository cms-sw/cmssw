import FWCore.ParameterSet.Config as cms

from ..modules.hltEG100EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltPhoton100EBTightIDTightIsoClusterShapeL1SeededFilter_cfi import *
from ..modules.hltPhoton100EBTightIDTightIsoEcalIsoL1SeededFilter_cfi import *
from ..modules.hltPhoton100EBTightIDTightIsoHcalIsoL1SeededFilter_cfi import *
from ..modules.hltPhoton100EBTightIDTightIsoHEL1SeededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..tasks.HLTPhoton100EBTightIDTightIsoOpenL1SeededTask_cfi import *

HLTPhoton100EBTightIDTightIsoOpenL1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG100EtL1SeededFilter +
    hltPhoton100EBTightIDTightIsoClusterShapeL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltPhoton100EBTightIDTightIsoHEL1SeededFilter +
    hltPhoton100EBTightIDTightIsoEcalIsoL1SeededFilter +
    HLTPFHcalClusteringForEgamma +
    hltPhoton100EBTightIDTightIsoHcalIsoL1SeededFilter,
    HLTPhoton100EBTightIDTightIsoOpenL1SeededTask
)

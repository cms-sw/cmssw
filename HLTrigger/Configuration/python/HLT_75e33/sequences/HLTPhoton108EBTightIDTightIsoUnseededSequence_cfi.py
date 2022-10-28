import FWCore.ParameterSet.Config as cms

from ..modules.hltEG108EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltEGL1SeedsForSinglePhotonIsolatedFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoClusterShapeUnseededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoEcalIsoUnseededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoHcalIsoUnseededFilter_cfi import *
from ..modules.hltPhoton108EBTightIDTightIsoHEUnseededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..tasks.HLTPhoton108EBTightIDTightIsoUnseededTask_cfi import *

HLTPhoton108EBTightIDTightIsoUnseededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForSinglePhotonIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltEG108EtUnseededFilter +
    hltPhoton108EBTightIDTightIsoClusterShapeUnseededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltPhoton108EBTightIDTightIsoHEUnseededFilter +
    hltPhoton108EBTightIDTightIsoEcalIsoUnseededFilter +
    HLTPFHcalClusteringForEgamma +
    hltPhoton108EBTightIDTightIsoHcalIsoUnseededFilter,
    HLTPhoton108EBTightIDTightIsoUnseededTask
)

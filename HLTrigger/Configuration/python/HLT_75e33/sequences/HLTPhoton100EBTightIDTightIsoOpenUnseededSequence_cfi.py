import FWCore.ParameterSet.Config as cms

from ..modules.hltEG100EtUnseededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperUnseeded_cfi import *
from ..modules.hltPhoton100EBTightIDTightIsoClusterShapeUnseededFilter_cfi import *
from ..modules.hltPhoton100EBTightIDTightIsoEcalIsoUnseededFilter_cfi import *
from ..modules.hltPhoton100EBTightIDTightIsoHcalIsoUnseededFilter_cfi import *
from ..modules.hltPhoton100EBTightIDTightIsoHEUnseededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..tasks.HLTPhoton100EBTightIDTightIsoOpenUnseededTask_cfi import *

HLTPhoton100EBTightIDTightIsoOpenUnseededSequence = cms.Sequence(
    HLTL1Sequence +
    HLTDoFullUnpackingEgammaEcalSequence +
    HLTPFClusteringForEgammaUnseeded +
    HLTHgcalTiclPFClusteringForEgammaUnseeded +
    hltEgammaCandidatesWrapperUnseeded +
    hltEG100EtUnseededFilter +
    cms.ignore(hltPhoton100EBTightIDTightIsoClusterShapeUnseededFilter) +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    cms.ignore(hltPhoton100EBTightIDTightIsoHEUnseededFilter) +
    cms.ignore(hltPhoton100EBTightIDTightIsoEcalIsoUnseededFilter) +
    HLTPFHcalClusteringForEgamma +
    cms.ignore(hltPhoton100EBTightIDTightIsoHcalIsoUnseededFilter),
    HLTPhoton100EBTightIDTightIsoOpenUnseededTask)

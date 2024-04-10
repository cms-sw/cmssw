import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTEle32WPTightUnseededInnerSequence_cfi import *
from ..sequences.HLTElePixelMatchUnseededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTGsfElectronUnseededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *

HLTEle32WPTightUnseededSequence = cms.Sequence(HLTL1Sequence
    +hltEGL1SeedsForSingleEleIsolatedFilter
    +HLTDoFullUnpackingEgammaEcalSequence
    +HLTEGammaDoLocalHcalSequence
    +HLTPFClusteringForEgammaUnseededSequence
    +HLTHgcalTiclPFClusteringForEgammaUnseededSequence
    +HLTFastJetForEgammaSequence
    +HLTPFHcalClusteringForEgammaSequence
    +HLTElePixelMatchUnseededSequence
    +HLTGsfElectronUnseededSequence
    +HLTEle32WPTightUnseededInnerSequence
)

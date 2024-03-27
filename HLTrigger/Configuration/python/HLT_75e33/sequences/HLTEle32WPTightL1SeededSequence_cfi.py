import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTEle32WPTightL1SeededInnerSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTGsfElectronL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *

HLTEle32WPTightL1SeededSequence = cms.Sequence(HLTL1Sequence
    +hltEGL1SeedsForSingleEleIsolatedFilter
    +HLTDoFullUnpackingEgammaEcalL1SeededSequence
    +HLTEGammaDoLocalHcalSequence
    +HLTPFClusteringForEgammaL1SeededSequence
    +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
    +HLTFastJetForEgammaSequence
    +HLTPFHcalClusteringForEgammaSequence
    +HLTElePixelMatchL1SeededSequence
    +HLTGsfElectronL1SeededSequence
    +HLTEle32WPTightL1SeededInnerSequence
)

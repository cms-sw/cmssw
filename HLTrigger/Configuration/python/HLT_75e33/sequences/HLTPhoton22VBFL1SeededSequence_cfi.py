import FWCore.ParameterSet.Config as cms

# Photon leg of the VBF + photon path: L1-seeded e/gamma reconstruction, ET > 22 GeV and H/E.
# No L1 e/gamma seed here; the path is seeded by the inclusive VBF L1T seed.
from ..modules.hltEG22EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltPhoton22VBFHEL1SeededFilter_cfi import *
from ..modules.hltPhoton22VBFHgcalHEL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *

HLTPhoton22VBFL1SeededSequence = cms.Sequence(
    HLTDoFullUnpackingEgammaEcalL1SeededSequence
    +HLTPFClusteringForEgammaL1SeededSequence
    +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
    +hltEgammaCandidatesL1Seeded
    +hltEgammaCandidatesWrapperL1Seeded
    +hltEG22EtL1SeededFilter
    +hltEgammaHGCALIDVarsL1Seeded
    +hltPhoton22VBFHgcalHEL1SeededFilter
    +HLTEGammaDoLocalHcalSequence
    +hltEgammaHoverEL1Seeded
    +HLTFastJetForEgammaSequence
    +hltPhoton22VBFHEL1SeededFilter
)

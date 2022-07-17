import FWCore.ParameterSet.Config as cms

from ..sequences.HLTAK4PFJetsReconstruction_cfi import *
from ..sequences.HLTAK8PFJetsReconstruction_cfi import *
from ..sequences.HLTCaloMETReconstruction_cfi import *
from ..sequences.HLTPFClusterJMEReconstruction_cfi import *
from ..sequences.HLTPFJetsCHSReconstruction_cfi import *
from ..sequences.HLTPFMETsReconstruction_cfi import *
from ..sequences.HLTPFPuppiJMEReconstruction_cfi import *
from ..sequences.HLTPFSoftKillerMETReconstruction_cfi import *

HLTJMESequence = cms.Sequence(
    HLTCaloMETReconstruction +
    HLTPFClusterJMEReconstruction +
    HLTAK4PFJetsReconstruction +
    HLTAK8PFJetsReconstruction +
    HLTPFJetsCHSReconstruction +
    HLTPFMETsReconstruction +
    HLTPFSoftKillerMETReconstruction +
    HLTPFPuppiJMEReconstruction
)

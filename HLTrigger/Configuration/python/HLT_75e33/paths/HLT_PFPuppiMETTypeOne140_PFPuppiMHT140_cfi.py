import FWCore.ParameterSet.Config as cms

from ..modules.hltPFPuppiMETTypeOne_cfi import *
from ..modules.hltPFPuppiMETTypeOne140_cfi import *
from ..modules.hltPFPuppiMETTypeOneCorrector_cfi import *
from ..modules.hltPFPuppiMHT_cfi import *
from ..modules.hltPFPuppiMHT140_cfi import *
from ..modules.l1tPFPuppiMET220off_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPFPuppiMETReconstruction_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

HLT_PFPuppiMETTypeOne140_PFPuppiMHT140 = cms.Path(
    HLTBeginSequence +
    l1tPFPuppiMET220off +
    HLTParticleFlowSequence +
    HLTAK4PFPuppiJetsReconstruction +
    HLTPFPuppiMETReconstruction +
    hltPFPuppiMETTypeOneCorrector +
    hltPFPuppiMETTypeOne +
    hltPFPuppiMETTypeOne140 +
    hltPFPuppiMHT +
    hltPFPuppiMHT140 +
    HLTEndSequence
)

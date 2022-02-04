import FWCore.ParameterSet.Config as cms

from ..modules.hltPFPuppiHT_cfi import *
from ..modules.hltPFPuppiMHT_cfi import *
from ..sequences.HLTJMESequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

MC_JME = cms.Path(
    HLTParticleFlowSequence +
    HLTJMESequence +
    hltPFPuppiHT +
    hltPFPuppiMHT
)

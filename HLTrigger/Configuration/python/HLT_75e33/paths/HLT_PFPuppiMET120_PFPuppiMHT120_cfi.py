import FWCore.ParameterSet.Config as cms

from ..modules.hltPFPuppiMET120_cfi import *
from ..modules.hltPFPuppiMHT_cfi import *
from ..modules.hltPFPuppiMHT120_cfi import *
from ..modules.l1tPFPuppiMET220off_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTPFPuppiMETReconstruction_cfi import *

HLT_PFPuppiMET120_PFPuppiMHT120 = cms.Path(l1tPFPuppiMET220off+HLTParticleFlowSequence+HLTPFPuppiMETReconstruction+hltPFPuppiMET120+HLTAK4PFPuppiJetsReconstruction+hltPFPuppiMHT+hltPFPuppiMHT120)

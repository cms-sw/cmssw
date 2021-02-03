import FWCore.ParameterSet.Config as cms

from ..modules.hltPFPuppiHT_cfi import *
from ..modules.hltPFPuppiHT1050_cfi import *
from ..modules.l1tPFPuppiHT_cfi import *
from ..modules.l1tPFPuppiHT450off_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

HLT_PFPuppiHT1050 = cms.Path(l1tPFPuppiHT+l1tPFPuppiHT450off+HLTParticleFlowSequence+HLTAK4PFPuppiJetsReconstruction+hltPFPuppiHT+hltPFPuppiHT1050)

import FWCore.ParameterSet.Config as cms

from ..modules.hltSingleAK4PFPuppiJet500_cfi import *
from ..modules.l1tSinglePFPuppiJet230off_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

HLT_AK4PFPuppiJet500 = cms.Path(l1tSinglePFPuppiJet230off+HLTParticleFlowSequence+HLTAK4PFPuppiJetsReconstruction+hltSingleAK4PFPuppiJet500)

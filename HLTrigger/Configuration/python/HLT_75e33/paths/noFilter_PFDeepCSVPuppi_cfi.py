import FWCore.ParameterSet.Config as cms

from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBtagDeepCSVSequencePFPuppi_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

noFilter_PFDeepCSVPuppi = cms.Path(HLTParticleFlowSequence+HLTAK4PFPuppiJetsReconstruction+HLTBtagDeepCSVSequencePFPuppi)

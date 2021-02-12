import FWCore.ParameterSet.Config as cms

from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBtagDeepFlavourSequencePFPuppi_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

noFilter_PFDeepFlavourPuppi = cms.Path(HLTParticleFlowSequence+HLTAK4PFPuppiJetsReconstruction+HLTBtagDeepFlavourSequencePFPuppi)

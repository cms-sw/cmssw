import FWCore.ParameterSet.Config as cms

from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBtagDeepCSVSequencePFPuppi_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTBtagDeepFlavourSequencePFPuppi_cfi import *

MC_BTV = cms.Path(HLTParticleFlowSequence+HLTAK4PFPuppiJetsReconstruction+HLTBtagDeepCSVSequencePFPuppi+HLTBtagDeepFlavourSequencePFPuppi)

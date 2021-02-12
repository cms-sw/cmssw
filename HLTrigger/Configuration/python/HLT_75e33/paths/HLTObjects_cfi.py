import FWCore.ParameterSet.Config as cms

from ..modules.hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4_cfi import *
from ..modules.hltPFPuppiHT_cfi import *
from ..modules.hltPFPuppiJetForBtagEta2p4_cfi import *
from ..modules.hltPFPuppiJetForBtagSelectorEta2p4_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

HLTObjects = cms.Path(HLTParticleFlowSequence+HLTAK4PFPuppiJetsReconstruction+hltPFPuppiHT+hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4+hltPFPuppiJetForBtagSelectorEta2p4+hltPFPuppiJetForBtagEta2p4)

import FWCore.ParameterSet.Config as cms

from ..modules.hltBTagPFPuppiDeepCSV0p865DoubleEta2p4_cfi import *
from ..modules.hltDoublePFPuppiJets128Eta2p4MaxDeta1p6_cfi import *
from ..modules.hltDoublePFPuppiJets128MaxEta2p4_cfi import *
from ..modules.l1tDoublePFPuppiJet112offMaxEta2p4_cfi import *
from ..modules.l1tDoublePFPuppiJets112offMaxDeta1p6_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTBtagDeepCSVSequencePFPuppiModEta2p4_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4 = cms.Path(
    HLTBeginSequence +
    l1tDoublePFPuppiJet112offMaxEta2p4 +
    l1tDoublePFPuppiJets112offMaxDeta1p6 +
    HLTParticleFlowSequence +
    HLTAK4PFPuppiJetsReconstruction +
    hltDoublePFPuppiJets128MaxEta2p4 +
    hltDoublePFPuppiJets128Eta2p4MaxDeta1p6 +
    HLTBtagDeepCSVSequencePFPuppiModEta2p4 +
    hltBTagPFPuppiDeepCSV0p865DoubleEta2p4 +
    HLTEndSequence
)

import FWCore.ParameterSet.Config as cms

from ..modules.hlt1PFPuppiCentralJet75MaxEta2p4_cfi import *
from ..modules.hlt2PFPuppiCentralJet60MaxEta2p4_cfi import *
from ..modules.hlt3PFPuppiCentralJet45MaxEta2p4_cfi import *
from ..modules.hlt4PFPuppiCentralJet40MaxEta2p4_cfi import *
from ..modules.hltBTagPFPuppiDeepCSV0p31Eta2p4TripleEta2p4_cfi import *
from ..modules.hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4_cfi import *
from ..modules.hltPFPuppiCentralJetQuad30MaxEta2p4_cfi import *
from ..modules.hltPFPuppiCentralJetsQuad30HT330MaxEta2p4_cfi import *
from ..modules.l1t1PFPuppiJet70offMaxEta2p4_cfi import *
from ..modules.l1t2PFPuppiJet55offMaxEta2p4_cfi import *
from ..modules.l1t4PFPuppiJet25OnlineMaxEta2p4_cfi import *
from ..modules.l1t4PFPuppiJet40offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHT400offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHT_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTBtagDeepCSVSequencePFPuppiModEta2p4_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepCSV_2p4 = cms.Path(
    HLTBeginSequence +
    l1tPFPuppiHT +
    l1tPFPuppiHT400offMaxEta2p4 +
    l1t1PFPuppiJet70offMaxEta2p4 +
    l1t2PFPuppiJet55offMaxEta2p4 +
    l1t4PFPuppiJet40offMaxEta2p4 +
    l1t4PFPuppiJet25OnlineMaxEta2p4 +
    HLTParticleFlowSequence +
    HLTAK4PFPuppiJetsReconstruction +
    hltPFPuppiCentralJetQuad30MaxEta2p4 +
    hlt1PFPuppiCentralJet75MaxEta2p4 +
    hlt2PFPuppiCentralJet60MaxEta2p4 +
    hlt3PFPuppiCentralJet45MaxEta2p4 +
    hlt4PFPuppiCentralJet40MaxEta2p4 +
    hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4 +
    hltPFPuppiCentralJetsQuad30HT330MaxEta2p4 +
    HLTBtagDeepCSVSequencePFPuppiModEta2p4 +
    hltBTagPFPuppiDeepCSV0p31Eta2p4TripleEta2p4 +
    HLTEndSequence
)

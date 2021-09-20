import FWCore.ParameterSet.Config as cms

from ..modules.hlt1PFPuppiCentralJet70MaxEta2p4_cfi import *
from ..modules.hlt2PFPuppiCentralJet40MaxEta2p4_cfi import *
from ..modules.hltBTagPFPuppiDeepFlavour0p375Eta2p4TripleEta2p4_cfi import *
from ..modules.hltPFPuppiCentralJetQuad30MaxEta2p4_cfi import *
from ..modules.hltPFPuppiCentralJetsQuad30HT200MaxEta2p4_cfi import *
from ..modules.l1t1PFPuppiJet70offMaxEta2p4_cfi import *
from ..modules.l1t2PFPuppiJet55offMaxEta2p4_cfi import *
from ..modules.l1t4PFPuppiJet25OnlineMaxEta2p4_cfi import *
from ..modules.l1t4PFPuppiJet40offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHT400offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHT_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTBtagDeepFlavourSequencePFPuppiModEta2p4_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4 = cms.Path(
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
    hlt1PFPuppiCentralJet70MaxEta2p4 +
    hlt2PFPuppiCentralJet40MaxEta2p4 +
    hltPFPuppiCentralJetsQuad30HT200MaxEta2p4 +
    HLTBtagDeepFlavourSequencePFPuppiModEta2p4 +
    hltBTagPFPuppiDeepFlavour0p375Eta2p4TripleEta2p4 +
    HLTEndSequence
)

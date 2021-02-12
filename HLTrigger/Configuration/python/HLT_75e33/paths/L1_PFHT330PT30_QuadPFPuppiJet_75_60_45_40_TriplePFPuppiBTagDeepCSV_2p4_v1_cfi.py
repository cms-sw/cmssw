import FWCore.ParameterSet.Config as cms

from ..modules.l1t1PFPuppiJet70offMaxEta2p4_cfi import *
from ..modules.l1t2PFPuppiJet55offMaxEta2p4_cfi import *
from ..modules.l1t4PFPuppiJet25OnlineMaxEta2p4_cfi import *
from ..modules.l1t4PFPuppiJet40offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHT400offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHTMaxEta2p4_cfi import *

L1_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepCSV_2p4_v1 = cms.Path(l1tPFPuppiHTMaxEta2p4+l1tPFPuppiHT400offMaxEta2p4+l1t1PFPuppiJet70offMaxEta2p4+l1t2PFPuppiJet55offMaxEta2p4+l1t4PFPuppiJet40offMaxEta2p4+l1t4PFPuppiJet25OnlineMaxEta2p4)

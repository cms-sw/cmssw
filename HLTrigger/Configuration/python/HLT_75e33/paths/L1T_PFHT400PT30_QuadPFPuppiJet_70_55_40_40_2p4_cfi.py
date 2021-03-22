import FWCore.ParameterSet.Config as cms

from ..modules.l1t1PFPuppiJet70offMaxEta2p4_cfi import *
from ..modules.l1t2PFPuppiJet55offMaxEta2p4_cfi import *
from ..modules.l1t4PFPuppiJet25OnlineMaxEta2p4_cfi import *
from ..modules.l1t4PFPuppiJet40offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHT400offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHT_cfi import *

L1T_PFHT400PT30_QuadPFPuppiJet_70_55_40_40_2p4 = cms.Path(l1tPFPuppiHT+l1tPFPuppiHT400offMaxEta2p4+l1t1PFPuppiJet70offMaxEta2p4+l1t2PFPuppiJet55offMaxEta2p4+l1t4PFPuppiJet40offMaxEta2p4+l1t4PFPuppiJet25OnlineMaxEta2p4)

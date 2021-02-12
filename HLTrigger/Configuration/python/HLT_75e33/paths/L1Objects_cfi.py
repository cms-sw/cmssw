import FWCore.ParameterSet.Config as cms

from ..modules.l1tDoublePFPuppiJet112offMaxEta2p4_cfi import *
from ..modules.l1tPFPuppiHTMaxEta2p4_cfi import *

L1Objects = cms.Path(l1tPFPuppiHTMaxEta2p4+l1tDoublePFPuppiJet112offMaxEta2p4)

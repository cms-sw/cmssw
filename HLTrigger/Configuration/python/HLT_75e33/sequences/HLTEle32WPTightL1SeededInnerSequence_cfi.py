import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6L1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *

HLTEle32WPTightL1SeededInnerSequence = cms.Sequence(hltEgammaHGCALIDVarsL1Seeded+hltEgammaEcalPFClusterIsoL1Seeded+hltEgammaHGCalLayerClusterIsoL1Seeded+hltEgammaHcalPFClusterIsoL1Seeded+hltEgammaEleL1TrkIsoL1Seeded+hltEgammaEleGsfTrackIsoV6L1Seeded)

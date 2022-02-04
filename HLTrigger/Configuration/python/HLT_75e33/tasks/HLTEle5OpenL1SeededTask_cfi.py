import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6L1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

HLTEle5OpenL1SeededTask = cms.Task(
    hltEgammaCandidatesL1Seeded,
    hltEgammaClusterShapeL1Seeded,
    hltEgammaEcalPFClusterIsoL1Seeded,
    hltEgammaEleGsfTrackIsoL1Seeded,
    hltEgammaEleGsfTrackIsoV6L1Seeded,
    hltEgammaEleL1TrkIsoL1Seeded,
    hltEgammaHGCALIDVarsL1Seeded,
    hltEgammaHGCalLayerClusterIsoL1Seeded,
    hltEgammaHcalPFClusterIsoL1Seeded,
    hltEgammaHoverEL1Seeded
)

import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaElectronPixelSeedsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *
from ..modules.hltEgammaPixelMatchVarsL1Seeded_cfi import *
from ..modules.hltEgammaSuperClustersToPixelMatchL1Seeded_cfi import *
from ..modules.hltElePixelHitDoubletsForTripletsL1Seeded_cfi import *
from ..modules.hltElePixelHitDoubletsL1Seeded_cfi import *
from ..modules.hltElePixelHitTripletsClusterRemoverL1Seeded_cfi import *
from ..modules.hltElePixelHitTripletsL1Seeded_cfi import *
from ..modules.hltElePixelSeedsCombinedL1Seeded_cfi import *
from ..modules.hltElePixelSeedsDoubletsL1Seeded_cfi import *
from ..modules.hltElePixelSeedsTripletsL1Seeded_cfi import *
from ..modules.hltEleSeedsTrackingRegionsL1Seeded_cfi import *
from ..modules.hltPixelLayerPairsL1Seeded_cfi import *
from ..modules.hltPixelLayerTriplets_cfi import *
from ..modules.MeasurementTrackerEvent_cfi import *

HLTElePixelMatchL1SeededTask = cms.Task(
    MeasurementTrackerEvent,
    hltEgammaElectronPixelSeedsL1Seeded,
    hltEgammaHoverEL1Seeded,
    hltEgammaPixelMatchVarsL1Seeded,
    hltEgammaSuperClustersToPixelMatchL1Seeded,
    hltElePixelHitDoubletsForTripletsL1Seeded,
    hltElePixelHitDoubletsL1Seeded,
    hltElePixelHitTripletsClusterRemoverL1Seeded,
    hltElePixelHitTripletsL1Seeded,
    hltElePixelSeedsCombinedL1Seeded,
    hltElePixelSeedsDoubletsL1Seeded,
    hltElePixelSeedsTripletsL1Seeded,
    hltEleSeedsTrackingRegionsL1Seeded,
    hltPixelLayerPairsL1Seeded,
    hltPixelLayerTriplets
)

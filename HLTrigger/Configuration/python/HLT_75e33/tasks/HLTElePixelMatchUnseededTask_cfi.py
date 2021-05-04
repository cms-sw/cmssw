import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaElectronPixelSeedsUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *
from ..modules.hltEgammaPixelMatchVarsUnseeded_cfi import *
from ..modules.hltEgammaSuperClustersToPixelMatchUnseeded_cfi import *
from ..modules.hltElePixelHitDoubletsForTripletsUnseeded_cfi import *
from ..modules.hltElePixelHitDoubletsUnseeded_cfi import *
from ..modules.hltElePixelHitTripletsClusterRemoverUnseeded_cfi import *
from ..modules.hltElePixelHitTripletsUnseeded_cfi import *
from ..modules.hltElePixelSeedsCombinedUnseeded_cfi import *
from ..modules.hltElePixelSeedsDoubletsUnseeded_cfi import *
from ..modules.hltElePixelSeedsTripletsUnseeded_cfi import *
from ..modules.hltEleSeedsTrackingRegionsUnseeded_cfi import *
from ..modules.hltPixelLayerPairsUnseeded_cfi import *
from ..modules.hltPixelLayerTriplets_cfi import *
from ..modules.MeasurementTrackerEvent_cfi import *

HLTElePixelMatchUnseededTask = cms.Task(
    MeasurementTrackerEvent,
    hltEgammaElectronPixelSeedsUnseeded,
    hltEgammaHoverEUnseeded,
    hltEgammaPixelMatchVarsUnseeded,
    hltEgammaSuperClustersToPixelMatchUnseeded,
    hltElePixelHitDoubletsForTripletsUnseeded,
    hltElePixelHitDoubletsUnseeded,
    hltElePixelHitTripletsClusterRemoverUnseeded,
    hltElePixelHitTripletsUnseeded,
    hltElePixelSeedsCombinedUnseeded,
    hltElePixelSeedsDoubletsUnseeded,
    hltElePixelSeedsTripletsUnseeded,
    hltEleSeedsTrackingRegionsUnseeded,
    hltPixelLayerPairsUnseeded,
    hltPixelLayerTriplets
)

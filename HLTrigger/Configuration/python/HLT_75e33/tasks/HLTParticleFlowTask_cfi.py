import FWCore.ParameterSet.Config as cms

from ..modules.generalTracks_cfi import *
from ..modules.hltPhase2PixelVertices_cfi import *
from ..modules.trackerClusterCheck_cfi import *
from ..tasks.HLTBeamSpotTask_cfi import *
from ..tasks.RawToDigiTask_cfi import *
from ..tasks.caloTowersRecTask_cfi import *
from ..tasks.ecalClustersTask_cfi import *
from ..tasks.hcalGlobalRecoTask_cfi import *
from ..tasks.hgcalLocalRecoTask_cfi import *
from ..tasks.highPtTripletStepTask_cfi import *
from ..tasks.highlevelrecoTask_cfi import *
from ..tasks.initialStepTask_cfi import *
from ..tasks.itLocalRecoTask_cfi import *
from ..tasks.iterTICLTask_cfi import *
from ..tasks.localrecoTask_cfi import *
from ..tasks.otLocalRecoTask_cfi import *
from ..tasks.particleFlowClusterTask_cfi import *
from ..tasks.hltPhase2PixelTracksTask_cfi import *
from ..tasks.vertexRecoTask_cfi import *

HLTParticleFlowTask = cms.Task(
    HLTBeamSpotTask,
    RawToDigiTask,
    caloTowersRecTask,
    ecalClustersTask,
    generalTracks,
    hcalGlobalRecoTask,
    hgcalLocalRecoTask,
    highPtTripletStepTask,
    highlevelrecoTask,
    hltOnlineBeamSpot,
    initialStepTask,
    itLocalRecoTask,
    iterTICLTask,
    localrecoTask,
    otLocalRecoTask,
    particleFlowClusterTask,
    hltPhase2PixelTracksTask,
    hltPhase2PixelVertices,
    trackerClusterCheck,
    vertexRecoTask
)

import FWCore.ParameterSet.Config as cms

from ..modules.generalTracks_cfi import *
from ..modules.offlineBeamSpot_cfi import *
from ..modules.pixelVertices_cfi import *
from ..modules.trackerClusterCheck_cfi import *
from ..tasks.caloTowersRecTask_cfi import *
from ..tasks.ecalClustersTask_cfi import *
from ..tasks.hcalGlobalRecoTask_cfi import *
from ..tasks.hgcalLocalRecoTask_cfi import *
from ..tasks.highlevelrecoTask_cfi import *
from ..tasks.highPtTripletStepTask_cfi import *
from ..tasks.initialStepTask_cfi import *
from ..tasks.iterTICLTask_cfi import *
from ..tasks.itLocalRecoTask_cfi import *
from ..tasks.localrecoTask_cfi import *
from ..tasks.otLocalRecoTask_cfi import *
from ..tasks.particleFlowClusterTask_cfi import *
from ..tasks.pixelTracksTask_cfi import *
from ..tasks.RawToDigiTask_cfi import *
from ..tasks.vertexRecoTask_cfi import *

HLTParticleFlowTask = cms.Task(
    RawToDigiTask,
    caloTowersRecTask,
    ecalClustersTask,
    generalTracks,
    hcalGlobalRecoTask,
    hgcalLocalRecoTask,
    highPtTripletStepTask,
    highlevelrecoTask,
    initialStepTask,
    itLocalRecoTask,
    iterTICLTask,
    localrecoTask,
    offlineBeamSpot,
    otLocalRecoTask,
    particleFlowClusterTask,
    pixelTracksTask,
    pixelVertices,
    trackerClusterCheck,
    vertexRecoTask
)

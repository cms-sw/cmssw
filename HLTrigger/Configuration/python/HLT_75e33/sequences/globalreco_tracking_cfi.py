import FWCore.ParameterSet.Config as cms

from ..modules.generalTracks_cfi import *
from ..modules.offlineBeamSpot_cfi import *
from ..modules.pixelVertices_cfi import *
from ..modules.trackerClusterCheck_cfi import *
from ..tasks.highPtTripletStepTask_cfi import *
from ..tasks.initialStepTask_cfi import *
from ..tasks.itLocalRecoTask_cfi import *
from ..tasks.otLocalRecoTask_cfi import *
from ..tasks.pixelTracksTask_cfi import *
from ..tasks.vertexRecoTask_cfi import *

globalreco_tracking = cms.Sequence(cms.Task(generalTracks, highPtTripletStepTask, initialStepTask, itLocalRecoTask, offlineBeamSpot, otLocalRecoTask, pixelTracksTask, pixelVertices, trackerClusterCheck, vertexRecoTask))

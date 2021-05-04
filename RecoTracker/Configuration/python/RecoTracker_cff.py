import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
# Iterative steps
from RecoTracker.IterativeTracking.iterativeTk_cff import *
from RecoTracker.IterativeTracking.ElectronSeeds_cff import *


import copy

#dEdX reconstruction
from RecoTracker.DeDx.dedxEstimators_cff import *

#BeamHalo tracking
from RecoTracker.Configuration.RecoTrackerBHM_cff import *


#special sequences, such as pixel-less
from RecoTracker.Configuration.RecoTrackerNotStandard_cff import *

ckftracks_woBHTask = cms.Task(iterTrackingTask,
                              electronSeedsSeqTask,
                              doAlldEdXEstimatorsTask)
ckftracks_woBH = cms.Sequence(ckftracks_woBHTask)
ckftracksTask = ckftracks_woBHTask.copy() #+ beamhaloTracksSeq) # temporarily out, takes too much resources
ckftracks = cms.Sequence(ckftracksTask) 

ckftracks_wodEdXTask = ckftracksTask.copy()
ckftracks_wodEdXTask.remove(doAlldEdXEstimatorsTask)
ckftracks_wodEdX = cms.Sequence(ckftracks_wodEdXTask)

ckftracks_plus_pixellessTask = cms.Task(ckftracksTask, ctfTracksPixelLessTask)
ckftracks_plus_pixelless = cms.Sequence(ckftracks_plus_pixellessTask)


from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackingGlobalRecoTask = cms.Task(ckftracksTask, trackExtrapolator)
trackingGlobalReco = cms.Sequence(trackingGlobalRecoTask)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(trackingGlobalRecoTask, cms.Task(doAlldEdXEstimatorsTask, trackExtrapolator))

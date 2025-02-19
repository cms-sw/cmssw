import FWCore.ParameterSet.Config as cms

# Iterative steps
from RecoTracker.IterativeTracking.iterativeTk_cff import *
from RecoTracker.IterativeTracking.ElectronSeeds_cff import *


# RS
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cff import *
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cff import *
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cff import *
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff import *

import copy

#dEdX reconstruction
from RecoTracker.DeDx.dedxEstimators_cff import *

#BeamHalo tracking
from RecoTracker.Configuration.RecoTrackerBHM_cff import *


#special sequences, such as pixel-less
from RecoTracker.Configuration.RecoTrackerNotStandard_cff import *

ckftracks_woBH = cms.Sequence(iterTracking*electronSeedsSeq*doAlldEdXEstimators)
ckftracks = ckftracks_woBH.copy() #+ beamhaloTracksSeq) # temporarily out, takes too much resources

ckftracks_wodEdX = ckftracks.copy()
ckftracks_wodEdX.remove(doAlldEdXEstimators)

rstracks = cms.Sequence(roadSearchSeeds*
                        roadSearchClouds*rsTrackCandidates*
                        rsWithMaterialTracks)

ckftracks_plus_pixelless = cms.Sequence(ckftracks*ctfTracksPixelLess)


from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackingGlobalReco = cms.Sequence(ckftracks*trackExtrapolator)

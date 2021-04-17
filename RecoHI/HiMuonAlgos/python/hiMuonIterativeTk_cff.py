import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedGenerator.trackerClusterCheck_cfi import trackerClusterCheck as _trackerClusterCheck
hiRegitMuClusterCheck = _trackerClusterCheck.clone(
    doClusterCheck = False # do not check for max number of clusters pixel or strips
)

from RecoHI.HiMuonAlgos.HiRegitMuonInitialStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonPixelPairStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonDetachedTripletStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonMixedTripletStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonPixelLessStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonSeededStep_cff import *

from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiGeneralAndRegitMuTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = [
	'hiRegitMuInitialStepTracks',
	'hiRegitMuPixelPairStepTracks',
	'hiRegitMuMixedTripletStepTracks',
	'hiRegitMuPixelLessStepTracks',
	'hiRegitMuDetachedTripletStepTracks',
	'hiRegitMuonSeededTracksOutIn',
	'hiRegitMuonSeededTracksInOut'
	],
    selectedTrackQuals = [
	"hiRegitMuInitialStepSelector:hiRegitMuInitialStepLoose",
	"hiRegitMuPixelPairStepSelector:hiRegitMuPixelPairStep",
	"hiRegitMuMixedTripletStepSelector:hiRegitMuMixedTripletStep",
	"hiRegitMuPixelLessStepSelector:hiRegitMuPixelLessStep",
	"hiRegitMuDetachedTripletStepSelector:hiRegitMuDetachedTripletStep",
	"hiRegitMuonSeededTracksOutInSelector:hiRegitMuonSeededTracksOutInHighPurity",
	"hiRegitMuonSeededTracksInOutSelector:hiRegitMuonSeededTracksInOutHighPurity"
        ],
    hasSelector = [1,1,1,1,1,1,1],
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6), pQual=cms.bool(True))),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
)

hiRegitMuTrackingTask = cms.Task(hiRegitMuClusterCheck
                                 ,hiRegitMuonInitialStepTask
                                 ,hiRegitMuonPixelPairStepTask
                                 ,hiRegitMuonMixedTripletStepTask
                                 ,hiRegitMuonPixelLessStepTask
                                 ,hiRegitMuonDetachedTripletStepTask
                                 ,hiRegitMuonSeededStepTask
                                 )
hiRegitMuTracking = cms.Sequence(hiRegitMuTrackingTask)

# Standalone muons
from RecoMuon.Configuration.RecoMuonPPonly_cff import *

hiRegitMuTrackingAndStaTask = cms.Task(standalonemuontrackingTask
                                 ,hiRegitMuTrackingTask)
hiRegitMuTrackingAndSta = cms.Sequence(hiRegitMuTrackingAndStaTask)

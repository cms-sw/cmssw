import FWCore.ParameterSet.Config as cms

from RecoHI.HiMuonAlgos.HiRegitMuonInitialStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonPixelPairStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonMixedTripletStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonPixelLessStep_cff import *

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiGeneralAndRegitMuTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('hiRegitMuInitialStepTracks'),
                      cms.InputTag('hiRegitMuPixelPairStepTracks'),
                      cms.InputTag('hiRegitMuMixedTripletStepTracks'),
                      cms.InputTag('hiRegitMuPixelLessStepTracks')
                      ),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiRegitMuInitialStepSelector","hiRegitMuInitialStepLoose"),
                                       cms.InputTag("hiRegitMuPixelPairStepSelector","hiRegitMuPixelPairStep"),
                                       cms.InputTag("hiRegitMuMixedTripletStepSelector","hiRegitMuMixedTripletStep"),
                                       cms.InputTag("hiRegitMuPixelLessStepSelector","hiRegitMuPixelLessStep")
                                       ),
    hasSelector=cms.vint32(1,1,1,1),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3), pQual=cms.bool(True))),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

hiRegitMuTracking = cms.Sequence(hiRegitMuonInitialStep
                                 *hiRegitMuonPixelPairStep
                                 *hiRegitMuonMixedTripletStep
                                 *hiRegitMuonPixelLessStep
                                 )

# Standalone muons
from RecoMuon.Configuration.RecoMuonPPonly_cff import *


hiRegitMuTrackingAndSta = cms.Sequence(standalonemuontracking
      *hiRegitMuTracking)


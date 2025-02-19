import FWCore.ParameterSet.Config as cms

from RecoHI.HiMuonAlgos.HiRegitMuonInitialStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonLowPtTripletStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonPixelPairStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonDetachedTripletStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonMixedTripletStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonPixelLessStep_cff import *
from RecoHI.HiMuonAlgos.HiRegitMuonTobTecStep_cff import *

from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
hiRegitMuGeneralTracks = RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff.generalTracks.clone(
    TrackProducers = (cms.InputTag('hiRegitMuInitialStepTracks'),
                      cms.InputTag('hiRegitMuLowPtTripletStepTracks'),
                      cms.InputTag('hiRegitMuPixelPairStepTracks'),
                      cms.InputTag('hiRegitMuDetachedTripletStepTracks'),
                      cms.InputTag('hiRegitMuMixedTripletStepTracks'),
                      cms.InputTag('hiRegitMuPixelLessStepTracks'),
                      cms.InputTag('hiRegitMuTobTecStepTracks')
                      ),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiRegitMuInitialStepSelector","hiRegitMuInitialStep"),
                                       cms.InputTag("hiRegitMuLowPtTripletStepSelector","hiRegitMuLowPtTripletStep"),
                                       cms.InputTag("hiRegitMuPixelPairStepSelector","hiRegitMuPixelPairStep"),
                                       cms.InputTag("hiRegitMuDetachedTripletStep"),
                                       cms.InputTag("hiRegitMuMixedTripletStep"),
                                       cms.InputTag("hiRegitMuPixelLessStepSelector","hiRegitMuPixelLessStep"),
                                       cms.InputTag("hiRegitMuTobTecStepSelector","hiRegitMuTobTecStep")
                                       )
    )

hiRegitMuonIterTracking = cms.Sequence(hiRegitMuonInitialStep*
                            hiRegitMuonLowPtTripletStep*
                            hiRegitMuonPixelPairStep*
                            hiRegitMuonDetachedTripletStep*
                            hiRegitMuonMixedTripletStep*
                            hiRegitMuonPixelLessStep*
                            hiRegitMuonTobTecStep*
                            hiRegitMuGeneralTracks
                            )





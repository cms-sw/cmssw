from RecoMuon.MuonIdentification.muonTrackExtraThinningProducer_cfi import muonTrackExtraThinningProducer
from RecoTracker.TrackProducer.trackingRecHitThinningProducer_cfi import trackingRecHitThinningProducer

import FWCore.ParameterSet.Config as cms

thinnedGeneralTrackExtras = muonTrackExtraThinningProducer.clone(inputTag = cms.InputTag("generalTracks"))

#standalone muons not needed here because full collection of both TrackExtras and TrackingRecHits are stored in AOD

thinnedGlobalMuonExtras = muonTrackExtraThinningProducer.clone(inputTag = cms.InputTag("globalMuons"))

thinnedTevMuonExtrasFirstHit = muonTrackExtraThinningProducer.clone(inputTag = cms.InputTag("tevMuons","firstHit"))

thinnedTevMuonExtrasPicky = muonTrackExtraThinningProducer.clone(inputTag = cms.InputTag("tevMuons","picky"))

thinnedTevMuonExtrasDyt = muonTrackExtraThinningProducer.clone(inputTag = cms.InputTag("tevMuons","dyt"))

thinnedGeneralTrackHits = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("generalTracks"),
                                                               trackExtraTag = cms.InputTag("thinnedGeneralTrackExtras"))

thinnedGlobalMuonHits = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("globalMuons"),
                                                               trackExtraTag = cms.InputTag("thinnedGlobalMuonExtras"))

thinnedTevMuonHitsFirstHit = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("tevMuons","firstHit"),
                                                               trackExtraTag = cms.InputTag("thinnedTevMuonExtrasFirstHit"))

thinnedTevMuonHitsPicky = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("tevMuons","picky"),
                                                               trackExtraTag = cms.InputTag("thinnedTevMuonExtrasPicky"))

thinnedTevMuonHitsDyt = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("tevMuons","dyt"),
                                                               trackExtraTag = cms.InputTag("thinnedTevMuonExtrasDyt"))

thinnedTrackExtrasTask = cms.Task(thinnedGeneralTrackExtras,
                                  thinnedGlobalMuonExtras,
                                  thinnedTevMuonExtrasFirstHit,
                                  thinnedTevMuonExtrasPicky,
                                  thinnedTevMuonExtrasDyt,
                                  thinnedGeneralTrackHits,
                                  thinnedGlobalMuonHits,
                                  thinnedTevMuonHitsFirstHit,
                                  thinnedTevMuonHitsPicky,
                                  thinnedTevMuonHitsDyt,
                                  )

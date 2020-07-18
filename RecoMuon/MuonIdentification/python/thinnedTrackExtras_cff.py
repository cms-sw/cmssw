from RecoMuon.MuonIdentification.muonTrackExtraThinningProducer_cfi import muonTrackExtraThinningProducer
from RecoTracker.TrackProducer.trackingRecHitThinningProducer_cfi import trackingRecHitThinningProducer
from RecoTracker.TrackProducer.siPixelClusterThinningProducer_cfi import siPixelClusterThinningProducer
from RecoTracker.TrackProducer.siStripClusterThinningProducer_cfi import siStripClusterThinningProducer

import FWCore.ParameterSet.Config as cms

thinnedGeneralTrackExtras = muonTrackExtraThinningProducer.clone(inputTag = cms.InputTag("generalTracks"),
                                                                  cut = cms.string("pt > 3. || isPFMuon"),
                                                                  slimTrajParams = cms.bool(True),
                                                                  slimResiduals = cms.bool(True),
                                                                  slimFinalState = cms.bool(True))

#standalone muons not needed here because full collection of both TrackExtras and TrackingRecHits are stored in AOD

thinnedGlobalMuonExtras = thinnedGeneralTrackExtras.clone(inputTag = cms.InputTag("globalMuons"))

thinnedTevMuonExtrasFirstHit = thinnedGeneralTrackExtras.clone(inputTag = cms.InputTag("tevMuons","firstHit"))

thinnedTevMuonExtrasPicky = thinnedGeneralTrackExtras.clone(inputTag = cms.InputTag("tevMuons","picky"))

thinnedTevMuonExtrasDyt = thinnedGeneralTrackExtras.clone(inputTag = cms.InputTag("tevMuons","dyt"))

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

thinnedSiPixelClusters = siPixelClusterThinningProducer.clone(inputTag = cms.InputTag("siPixelClusters"),
                                                              trackingRecHitsTags=cms.VInputTag("thinnedGeneralTrackHits",
                                                                                    "thinnedGlobalMuonHits",
                                                                                    "thinnedTevMuonHitsFirstHit",
                                                                                    "thinnedTevMuonHitsPicky",
                                                                                    "thinnedTevMuonHitsDyt"))

thinnedSiStripClusters = siStripClusterThinningProducer.clone(inputTag = cms.InputTag("siStripClusters"),
                                                              trackingRecHitsTags=cms.VInputTag("thinnedGeneralTrackHits",
                                                                                    "thinnedGlobalMuonHits",
                                                                                    "thinnedTevMuonHitsFirstHit",
                                                                                    "thinnedTevMuonHitsPicky",
                                                                                    "thinnedTevMuonHitsDyt"))

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
                                  thinnedSiPixelClusters,
                                  thinnedSiStripClusters,
                                  )

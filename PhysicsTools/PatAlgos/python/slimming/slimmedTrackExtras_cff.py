from RecoMuon.MuonIdentification.muonTrackExtraThinningProducer_cfi import muonTrackExtraThinningProducer
from RecoTracker.TrackProducer.trackingRecHitThinningProducer_cfi import trackingRecHitThinningProducer
from RecoTracker.TrackProducer.siPixelClusterThinningProducer_cfi import siPixelClusterThinningProducer
from RecoTracker.TrackProducer.siStripClusterThinningProducer_cfi import siStripClusterThinningProducer

import FWCore.ParameterSet.Config as cms

slimmedGeneralTrackExtras = muonTrackExtraThinningProducer.clone(inputTag = cms.InputTag("thinnedGeneralTrackExtras"),
                                                  muonTag = "slimmedMuons",
                                                  cut = "",
                                                  )

#this one points to the original full collection of TrackExtras since it is available in the AOD
slimmedStandAloneMuonExtras = slimmedGeneralTrackExtras.clone(inputTag = cms.InputTag("standAloneMuons"))

slimmedGlobalMuonExtras = slimmedGeneralTrackExtras.clone(inputTag = cms.InputTag("thinnedGlobalMuonExtras"))

slimmedTevMuonExtrasFirstHit = slimmedGeneralTrackExtras.clone(inputTag = cms.InputTag("thinnedTevMuonExtrasFirstHit"))

slimmedTevMuonExtrasPicky = slimmedGeneralTrackExtras.clone(inputTag = cms.InputTag("thinnedTevMuonExtrasPicky"))

slimmedTevMuonExtrasDyt = slimmedGeneralTrackExtras.clone(inputTag = cms.InputTag("thinnedTevMuonExtrasDyt"))

slimmedGeneralTrackHits = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("thinnedGeneralTrackHits"),
                                                               trackExtraTag = cms.InputTag("slimmedGeneralTrackExtras"))

#this one points to the original full collection of TrackingRecHits since it is available in the AOD
slimmedStandAloneMuonHits = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("standAloneMuons"),
                                                               trackExtraTag = cms.InputTag("slimmedStandAloneMuonExtras"))

slimmedGlobalMuonHits = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("thinnedGlobalMuonHits"),
                                                               trackExtraTag = cms.InputTag("slimmedGlobalMuonExtras"))

slimmedTevMuonHitsFirstHit = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("thinnedTevMuonHitsFirstHit"),
                                                               trackExtraTag = cms.InputTag("slimmedTevMuonExtrasFirstHit"))

slimmedTevMuonsHitsPicky = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("thinnedTevMuonHitsPicky"),
                                                               trackExtraTag = cms.InputTag("slimmedTevMuonExtrasPicky"))

slimmedTevMuonHitsDyt = trackingRecHitThinningProducer.clone(inputTag = cms.InputTag("thinnedTevMuonHitsDyt"),
                                                               trackExtraTag = cms.InputTag("slimmedTevMuonExtrasDyt"))

slimmedSiPixelClusters = siPixelClusterThinningProducer.clone(inputTag = cms.InputTag("thinnedSiPixelClusters"),
                                                              trackingRecHitsTags=cms.VInputTag("slimmedGeneralTrackHits",
                                                                                    "slimmedGlobalMuonHits",
                                                                                    "slimmedTevMuonHitsFirstHit",
                                                                                    "slimmedTevMuonsHitsPicky",
                                                                                    "slimmedTevMuonHitsDyt"))
                                                              
slimmedSiStripClusters = siStripClusterThinningProducer.clone(inputTag = cms.InputTag("thinnedSiStripClusters"),
                                                              trackingRecHitsTags=cms.VInputTag("slimmedGeneralTrackHits",
                                                                                    "slimmedGlobalMuonHits",
                                                                                    "slimmedTevMuonHitsFirstHit",
                                                                                    "slimmedTevMuonsHitsPicky",
                                                                                    "slimmedTevMuonHitsDyt"))

slimmedTrackExtrasTask = cms.Task(slimmedGeneralTrackExtras,
                                  slimmedStandAloneMuonExtras,
                                  slimmedGlobalMuonExtras,
                                  slimmedTevMuonExtrasFirstHit,
                                  slimmedTevMuonExtrasPicky,
                                  slimmedTevMuonExtrasDyt,
                                  slimmedGeneralTrackHits,
                                  slimmedStandAloneMuonHits,
                                  slimmedGlobalMuonHits,
                                  slimmedTevMuonHitsFirstHit,
                                  slimmedTevMuonsHitsPicky,
                                  slimmedTevMuonHitsDyt,
                                  slimmedSiPixelClusters,
                                  slimmedSiStripClusters,
                                  )

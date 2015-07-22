import FWCore.ParameterSet.Config as cms

photonIDValueMapProducer = cms.EDProducer('PhotonIDValueMapProducer',
                                          # The module automatically detects AOD vs miniAOD, so we configure both
                                          #
                                          # AOD case
                                          #
                                          ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                          eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                          esReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsES"),
                                          particleBasedIsolation = cms.InputTag("particleBasedIsolation","gedPhotons"),
                                          vertices = cms.InputTag("offlinePrimaryVertices"),
                                          pfCandidates = cms.InputTag("particleFlow"),
                                          src = cms.InputTag('gedPhotons'),
                                          #
                                          # miniAOD case
                                          #
                                          ebReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedEBRecHits"),
                                          eeReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedEERecHits"),
                                          esReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedESRecHits"),
                                          verticesMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                          pfCandidatesMiniAOD = cms.InputTag("packedPFCandidates"),
                                          # there is no need for the isolation map here, for miniAOD it is inside packedPFCandidates
                                          srcMiniAOD = cms.InputTag('slimmedPhotons',processName=cms.InputTag.skipCurrentProcess()),
                                          )

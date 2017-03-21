import FWCore.ParameterSet.Config as cms

heepIDVarValueMaps = cms.EDProducer("ElectronHEEPIDValueMapProducer",
                                    beamSpot=cms.InputTag("offlineBeamSpot"),
                                    ebRecHitsAOD=cms.InputTag("reducedEcalRecHitsEB"),
                                    eeRecHitsAOD=cms.InputTag("reducedEcalRecHitsEB"),
                                    candsAOD=cms.VInputTag("packedCandsForTkIso",
                                                           "lostTracksForTkIso"),
                                    elesAOD=cms.InputTag("gedGsfElectrons"),
                                    ebRecHitsMiniAOD=cms.InputTag("reducedEgamma","reducedEBRecHits"),
                                    eeRecHitsMiniAOD=cms.InputTag("reducedEgamma","reducedEERecHits"),
                                    candsMiniAOD=cms.VInputTag("packedPFCandidates",
                                                               "lostTracks"),
                                    elesMiniAOD=cms.InputTag("slimmedElectrons"),
                                    dataFormat=cms.int32(0),#0 = auto detection, 1 = AOD, 2 = miniAOD

                                    trkIsoConfig= cms.PSet(
                                       barrelCuts=cms.PSet(
                                          minPt=cms.double(1.0),
                                          maxDR=cms.double(0.3),
                                          minDR=cms.double(0.0),
                                          minDEta=cms.double(0.005),
                                          maxDZ=cms.double(0.1),
                                          maxDPtPt=cms.double(0.1),
                                          minHits=cms.int32(8),
                                          minPixelHits=cms.int32(1),
                                          allowedQualities=cms.vstring(),
                                          algosToReject=cms.vstring()
                                          ),
                                       endcapCuts=cms.PSet(
                                          minPt=cms.double(1.0),
                                          maxDR=cms.double(0.3),
                                          minDR=cms.double(0.0),
                                          minDEta=cms.double(0.005),
                                          maxDZ=cms.double(0.5),
                                          maxDPtPt=cms.double(0.1),
                                          minHits=cms.int32(8),
                                          minPixelHits=cms.int32(1),
                                          allowedQualities=cms.vstring(),
                                          algosToReject=cms.vstring()
                                          )
                                       )
                                    )

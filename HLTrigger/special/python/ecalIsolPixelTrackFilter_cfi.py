import FWCore.ParameterSet.Config as cms

#HLTPixelIsolTrackFilter configuration
ecalIsolPixelTrackFilter = cms.EDFilter("HLTEcalPixelIsolTrackFilter",
                                        MaxEnergyIn = cms.double(10.0),
                                        MaxEnergyOut = cms.double(10.0),
                                        candTag = cms.InputTag("isolEcalPixelTrackProd"),
                                        NMaxTrackCandidates=cms.int32(10),
                                        DropMultiL2Event = cms.bool(False),
                                        saveTags = cms.bool( False )
                                        )



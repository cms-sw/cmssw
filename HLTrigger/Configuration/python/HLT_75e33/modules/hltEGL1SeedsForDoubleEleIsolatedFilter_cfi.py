import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForDoubleEleIsolatedFilter = cms.EDFilter("HLTP2GTDoubleObjectFilter",
                                                      l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
                                                      l1GTAlgos = cms.VPSet(
                                                          cms.PSet(
                                                              name = cms.string("pIsoTkEleEGEle22_12"),
                                                              collection1 = cms.PSet(
                                                                  objectType = cms.string("CL2Electrons"),
                                                                  minPt      = cms.double(0.),
                                                                  maxAbsEta  = cms.double(9999.),
                                                              ),
                                                              collection2 = cms.PSet(
                                                                  objectType = cms.string("CL2Photons"),
                                                                  minPt      = cms.double(0.),
                                                                  maxAbsEta  = cms.double(9999.),
                                                              ),
                                                              minDR      = cms.double(0.0),
                                                              maxDR      = cms.double(1e9),
                                                              minDEta    = cms.double(-1.0),
                                                              minDPhi    = cms.double(-1.0),
                                                              minInvMass = cms.double(0.0),
                                                              maxInvMass = cms.double(1e9),
                                                          ),
                                                          cms.PSet(
                                                              name = cms.string("pDoubleEGEle37_24"),
                                                              collection1 = cms.PSet(
                                                                  objectType = cms.string("CL2Photons"),
                                                                  minPt      = cms.double(0.),
                                                                  maxAbsEta  = cms.double(9999.),
                                                              ),
                                                              collection2 = cms.PSet(
                                                                  objectType = cms.string("CL2Photons"),
                                                                  minPt      = cms.double(0.),
                                                                  maxAbsEta  = cms.double(9999.),
                                                              ),
                                                              minDR      = cms.double(0.0),
                                                              maxDR      = cms.double(1e9),
                                                              minDEta    = cms.double(-1.0),
                                                              minDPhi    = cms.double(-1.0),
                                                              minInvMass = cms.double(0.0),
                                                              maxInvMass = cms.double(1e9),
                                                          ),
                                                          cms.PSet(
                                                              name = cms.string("pDoubleTkEle25_12"),
                                                              collection1 = cms.PSet(
                                                                  objectType = cms.string("CL2Electrons"),
                                                                  minPt      = cms.double(0.),
                                                                  maxAbsEta  = cms.double(9999.),
                                                              ),
                                                              collection2 = cms.PSet(
                                                                  objectType = cms.string("CL2Electrons"),
                                                                  minPt      = cms.double(0.),
                                                                  maxAbsEta  = cms.double(9999.),
                                                              ),
                                                              minDR      = cms.double(0.0),
                                                              maxDR      = cms.double(1e9),
                                                              minDEta    = cms.double(-1.0),
                                                              minDPhi    = cms.double(-1.0),
                                                              minInvMass = cms.double(0.0),
                                                              maxInvMass = cms.double(1e9),
                                                          ),
                                                      )
                                                    )
